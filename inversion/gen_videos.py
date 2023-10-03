# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm
import mrcfile

import legacy

from camera_utils import LookAtPoseSampler
from torch_utils import misc

from configs import global_config, paths_config, hyperparameters
import pickle
import json
import random

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

# ---------------------------- copied from mystyle --------------------------- #
import sys
module_path = '/playpen-nas-ssd/luchao/projects/mystyle'
if module_path not in sys.path:
    sys.path.append(module_path)
from utils_copy import id_utils # make a copy of utils to avoid import error as eg3d also has utils
id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)

# --------------------------- video config options --------------------------- #
from gen_video_config import evaluation, latent_interpolation

import PIL.Image
def save_image_grid(img, drange, grid_size):
    # save using PIL.Image.fromarray(img, 'RGB').save(fname)
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])
    return img
#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

def gen_interp_video(G, ws, mp4: str, cam=None, outdir=None, seeds=None, shuffle_seed=None, w_frames=120, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, device=torch.device('cuda'), **video_kwargs):

    # copied from 3dgan-inv

    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    num_keyframes = 1
    # * frontal view
    camera_lookat_point = torch.tensor([0, 0, 0], device=device)
    # camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device) # based on the average camera pivot
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(500, 1)
    _ = G.synthesis(ws[:1], c[:1]) # warm up

    # outdir = './out'
    id_name_ = paths_config.input_id # with png extension
    id_name = os.path.splitext(id_name_)[0] # without extension
    model_name = paths_config.eg3d_ffhq.split('/')[-1]
    model_name = os.path.splitext(model_name)[0]

    if evaluation and image_mode == 'image':
        # ------------------------- reconstruction evaluation ------------------------ #

        # ----------------------------------- lpips ---------------------------------- #
        dataset_json = paths_config.dataset_json
        with open(dataset_json, 'r') as f:
            cameras = json.load(f)
            cameras = {k: v for k, v in cameras['labels']}
        camera = cameras[id_name_]
        camera = torch.tensor(camera, device=device).unsqueeze(0)
        w_ = ws.squeeze().unsqueeze(0) # torch.Size([1, 14, 512])
        output_img = G.synthesis(ws=w_, c=camera, noise_mode='const')[image_mode].to(device) # [1, 3, 512, 512] [B, C, H, W]
        input_img = PIL.Image.open(os.path.join(paths_config.input_data_path, f'{id_name}.png')).convert('RGB') # [512, 512, 3] [H, W, C]
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        input_img = transform(input_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]

        # lpips requires normalized images
        from lpips import LPIPS
        lpips = LPIPS(net='alex').to(device).eval()
        lpips_loss = lpips(input_img, output_img)
        lpips_loss = lpips_loss.squeeze().item()
        lpips_loss = round(lpips_loss, 3)

        # ----------------------------------- psnr ----------------------------------- #
        # psnr requires denormalized images
        input_img_denorm = denorm(input_img)
        output_img_denorm = denorm(output_img)
        # type casting to uint8 then back to float32 is necessary for psnr calculation
        input_img_denorm = input_img_denorm.to(torch.uint8).to(torch.float32)
        output_img_denorm = output_img_denorm.to(torch.uint8).to(torch.float32)
        mse = torch.mean((input_img_denorm - output_img_denorm) ** 2)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        psnr = psnr.squeeze().item()
        psnr = round(psnr, 3)

        # ----------------------------------- DISTS ---------------------------------- #
        # DISTS requires normalized images
        from DISTS_pytorch import DISTS
        dists = DISTS().to(device)
        dists_loss = dists(input_img, output_img)
        dists_loss = dists_loss.squeeze().item()
        dists_loss = round(dists_loss, 3)

        # --------------------------------- id_error --------------------------------- #
        # # id_error requires denormalized images
        input_img_denorm = denorm(input_img)
        output_img_denorm = denorm(output_img)
        input_feature = person_identifier.get_feature(input_img_denorm.squeeze())
        output_feature = person_identifier.get_feature(output_img_denorm.squeeze())
        sim = person_identifier.compute_similarity(input_feature, output_feature)
        sim = sim.item()
        sim = round(sim, 3)
        id_error = 1 - sim
        
        # save synthesized image for sanity check
        img_to_save = save_image_grid(torch.cat([input_img.clone().detach().cpu(), output_img.clone().detach().cpu()]), drange=[-1, 1], grid_size=(2, 1)) # img_to_save:  (512, 1024, 3)
        # PIL.Image.fromarray(img_to_save).save(os.path.join(outdir, f'reconstruction_{id_name}_{model_name}_{dists_loss}_{lpips_loss}_{psnr}.png'))
        PIL.Image.fromarray(img_to_save).save(os.path.join(outdir, f'reconstruction_{id_name}_{model_name}_{dists_loss}_{lpips_loss}_{psnr}.pdf'))
    
    # run latent interpolation with given latent code and camera pose
    if latent_interpolation and image_mode == 'image':
        # ----------------------------------- latent interpolation ---------------------------------- #
        w_ = ws.squeeze().unsqueeze(0) # torch.Size([1, 14, 512])
        cam_ = cam.squeeze().unsqueeze(0) # torch.Size([1, 25])
        output_img = G.synthesis(ws=w_, c=cam_, noise_mode='const')[image_mode].to(device) # [1, 3, 512, 512] [B, C, H, W]
        img_to_save = save_image_grid(output_img.clone().detach().cpu(), drange=[-1, 1], grid_size=(1, 1)) # img_to_save:  (512, 512, 3)
        # cur_time = time.strftime("%M%S", time.localtime())
        # img_to_save_name = f'latent_interpolation_{model_name}_{cur_time}.png'
        # PIL.Image.fromarray(img_to_save).save(os.path.join(outdir, f'interpolation.png'))
        PIL.Image.fromarray(img_to_save).save(os.path.join(outdir, f'interpolation.pdf'))


    ws = ws.reshape(grid_h, grid_w, *ws.shape[1:]).unsqueeze(2).repeat(1,1,num_keyframes,1,1)

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    max_batch = 10000000
    voxel_resolution = 1024
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    all_poses = []
    images_out_interpolate = []

    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                        camera_lookat_point, radius=2.7, device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # (1, 25)
                c = c.type(torch.float32)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                w = w.type(torch.float32)  # ! important to cast to float32
                
                entangle = 'camera'
                if entangle == 'conditioning':
                    c_forward = torch.cat([LookAtPoseSampler.sample(3.14/2,
                                                                    3.14/2,
                                                                    camera_lookat_point,
                                                                    radius=2.7, device=device).reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c_forward, noise_mode='const')[image_mode][0]
                elif entangle == 'camera':
                    img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], noise_mode='const')[image_mode][0] # torch.Size([3, 512, 512])
                    # ! 存一下low resolution的图像
                    # print(w.unsqueeze(0).shape, c[0:1].shape)
                    # torch.Size([1, 14, 512]) torch.Size([1, 25])
                elif entangle == 'both':
                    w_c = G.mapping(z=zs[0:1], c=c[0:1], truncation_psi=psi, truncation_cutoff=truncation_cutoff)
                    img = G.synthesis(ws=w_c, c=c[0:1], noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)
        
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h)) # layout_grid().shape: (512, 512, 3)

        # # calculate id_error for each frame compared to the input image
        # if evaluation and (image_mode == 'image' or image_mode == 'image_depth'):
        #     input_img = PIL.Image.open(os.path.join(paths_config.input_data_path, f'{id_name}.png')).convert('RGB') 
        #     input_img = np.array(input_img) # (512, 512, 3)
        #     output_img = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h) 
        #     output_img = np.array(output_img) # (512, 512, 3)
        #     if image_mode == 'image':
        #         input_feature = person_identifier.get_feature(input_img)
        #         output_feature = person_identifier.get_feature(output_img)
        #         sim = person_identifier.compute_similarity(input_feature, output_feature)
        #         sim = sim.item()
        #         all_id_error.append(1-sim)

        # save images
        if image_mode == 'image' or image_mode == 'image_depth':
            output_img = layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h) 
            output_img = np.array(output_img) # (512, 512, 3)
            images_out_interpolate.append(output_img)

    video_out.close()
    all_poses = np.stack(all_poses)

    # calculate id_error for random frames compared to the input image
    if evaluation and image_mode == 'image':
        # id features for all input images
        run_name = paths_config.run_name
        reference_dir = '/playpen-nas-ssd/luchao/projects/eg3d/eg3d/dataset/{}'.format(run_name)
        imgs = os.listdir(reference_dir)
        imgs = [os.path.join(reference_dir, img) for img in imgs if img.endswith('.png')] # also consider mirrored images
        id_feature_dict = {}
        for img_path in imgs:
            img = PIL.Image.open(img_path).convert('RGB')
            img = np.array(img)
            feature = person_identifier.get_feature(img)
            id_feature_dict[img_path] = feature

        # randomly sample images from images_out_interpolate
        N = len(images_out_interpolate)
        n = N // 5 # ~20 frames
        indices = np.random.choice(N, n, replace=False)
        images_selected = [images_out_interpolate[i] for i in indices]

        # calculate similarities for all generated images vs. reference images
        id_errors = []
        for img in images_selected:
            feature = person_identifier.get_feature(img)
            # closest neighbor from the input images
            sims = {k: person_identifier.compute_similarity(feature, v) for k, v in id_feature_dict.items()}
            # get the clostest neighbor - largest similarity
            sim = max(sims.values()).item()
            # id score is 1-complement of the distance
            id_errors.append(1-sim)
        id_error_mean = round(np.mean(id_errors), 3)

    # save N images for video interpolation
    if evaluation and (image_mode == 'image' or image_mode == 'image_depth'):
        n = 7
        if image_mode == 'image':
            indices = np.linspace(0, len(images_out_interpolate)-1, n).astype(int)[1:-1]
            images_out = [images_out_interpolate[i] for i in indices]

            # put input image at the beginning
            input_img = PIL.Image.open(os.path.join(paths_config.input_data_path, f'{id_name}.png')).convert('RGB') 
            input_img = np.array(input_img) # (512, 512, 3)
            images_out = [input_img] + images_out
            
            # stack horizontally
            images_out = np.hstack(images_out)
            # save
            # PIL.Image.fromarray(images_out).save(os.path.join(outdir, 'interpolate_video_{}_{}_{}_{}.png'.format(id_name, model_name, image_mode, id_error_mean)))
            PIL.Image.fromarray(images_out).save(os.path.join(outdir, 'interpolate_video_{}_{}_{}_{}.pdf'.format(id_name, model_name, image_mode, id_error_mean)))
        else: # image_mode == 'image_depth'
            indices = np.linspace(0, len(images_out_interpolate)-1, n).astype(int)[1:-1]
            images_out = [images_out_interpolate[i] for i in indices]
            images_out = np.hstack(images_out)
            # squeeze
            images_out = np.squeeze(images_out, axis=-1)
            # PIL.Image.fromarray(images_out, mode='L').save(os.path.join(outdir, 'interpolate_video_{}_{}_{}.png'.format(id_name, model_name, image_mode)))
            PIL.Image.fromarray(images_out, mode='L').save(os.path.join(outdir, 'interpolate_video_{}_{}_{}.pdf'.format(id_name, model_name, image_mode)))

    if gen_shapes:
    # if evaluation and image_mode == 'image':
        with open(mp4.replace('.mp4', '_trajectory_{}_{}.npy'.format(id_name, model_name)), 'wb') as f:
            np.save(f, all_poses)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Output directory', type=str, required=True, metavar='DIR')
@click.option('--reload_modules', help='reload tuned_G', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--cfg', help='Config', type=click.Choice(['FFHQ', 'AFHQ', 'Shapenet']), required=False, metavar='STR', default='FFHQ', show_default=True)
@click.option('--image_mode', help='Image mode', type=click.Choice(['image', 'image_depth', 'image_raw']), required=False, metavar='STR', default='image', show_default=True)
@click.option('--sample_mult', 'sampling_multiplier', type=float, help='Multiplier for depth sampling in volume rendering', default=2, show_default=True)
@click.option('--nrr', type=int, help='Neural rendering resolution override', default=None, show_default=True)
@click.option('--shapes', type=bool, help='Gen shapes for shape interpolation', default=False, show_default=True)
@click.option('--interpolate', type=bool, help='Interpolate between seeds', default=True, show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    truncation_cutoff: int,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    outdir: str,
    reload_modules: bool,
    cfg: str,
    image_mode: str,
    sampling_multiplier: float,
    nrr: Optional[int],
    shapes: bool,
    interpolate: bool,
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    # ---------------------------------- network --------------------------------- #
    network_pkl = paths_config.eg3d_ffhq if reload_modules else network_pkl
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if reload_modules:
        new_G_path = f'{paths_config.checkpoints_dir}/tuned_G.pt'
        G.load_state_dict(torch.load(new_G_path))
    
    # revert reloading G to reload w only, basically with/without PTI
    if latent_interpolation:
        network_pkl = paths_config.eg3d_ffhq
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)

    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    print('Generator - rendering metrics:')
    print('depth resolution: ',G.rendering_kwargs['depth_resolution'])
    print('depth resolution importance: ', G.rendering_kwargs['depth_resolution_importance'])
    print('neural rendering resolution: ', G.neural_rendering_resolution)

    # ---------------------------- latent_code/camera ---------------------------- #
    camera_lookat_point = torch.tensor(G.rendering_kwargs['avg_camera_pivot'], device=device)
    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius'], device=device)
    focal_length = 4.2647 if cfg != 'Shapenet' else 1.7074 # shapenet has higher FOV
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(len(zs), 1)
    # convert c to double
    c = c.type(torch.float32)
    ws = G.mapping(z=zs, c=c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
    cam=None
    if reload_modules or latent_interpolation:
        # reload tuned latent [1, 14, 512]
        import pickle
        ws_path = f'{paths_config.checkpoints_dir}/w_pivot.pkl'
        ws = torch.from_numpy(pickle.load(open(ws_path, 'rb'))).to(device) # [1, 14, 512]
        cam_path = f'{paths_config.checkpoints_dir}/cam_pivot.pkl'
        cam = torch.from_numpy(pickle.load(open(cam_path, 'rb'))).unsqueeze(0).to(device) # [1, 25]
        cam = cam.type(torch.float32)

    if interpolate:
        output = os.path.join(outdir, 'interpolation.mp4')
        gen_interp_video(G=G, ws=ws, cam=cam, outdir=outdir, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode, gen_shapes=shapes)
    else:
        for seed in seeds:
            output = os.path.join(outdir, f'{seed}.mp4')
            seeds_ = [seed]
            gen_interp_video(G=G, ws=ws, mp4=output, bitrate='10M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds_, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode)

    # rename the output video
    # model_name = network_pkl.split('/')[-1].split('.')[0]
    model_name = os.path.splitext(os.path.basename(network_pkl))[0]
    if reload_modules:
        id_name = paths_config.input_id
        id_name = os.path.splitext(os.path.basename(id_name))[0]
        out_name = f'interpolation_{id_name}_final_{model_name}.mp4'
        if image_mode == 'image_depth':
            out_name = f'interpolation_{id_name}_depth_{model_name}.mp4'
        elif image_mode == 'image_raw':
            out_name = f'interpolation_{id_name}_raw_{model_name}.mp4'
        cmd = f'mv {outdir}/interpolation.mp4 {outdir}/{out_name}'
        os.system(cmd)
    # else:
    #     cmd = f'mv {outdir}/interpolation.mp4 {outdir}/interpolation.mp4'
    #     os.system(cmd)
    # cmd = 'python gen_videos_orig.py --outdir=out --trunc=0.7 --seeds=0-3'
    # os.system(cmd)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

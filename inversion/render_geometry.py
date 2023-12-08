import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import os.path as osp
import trimesh
import mcubes
import pyrender
from pyglet import gl
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from tqdm import tqdm
from PIL import Image
import glob
import imageio
import cv2
import click
import pdb
import random
from camera_utils import LookAtPoseSampler
import PIL.Image
from configs import paths_config
import json
# from training.volumetric_rendering import sample_camera_positions, create_cam2world_matrix

"""From https://github.com/MrTornado24/IDE-3D/blob/main/training/volumetric_rendering.py"""
def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=0.3, vertical_stddev=0.155, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = (torch.rand((n, 1), device=device) - .5) * 2 * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = ((torch.rand((n,1), device=device) - .5) * 2 * v_stddev + v_mean)
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

@click.command()
@click.option('--fname', type=str, help='Network pickle filename', required=True)
@click.option('--size', type=int, help='Size of mesh', default=512, required=False)
@click.option('--sigma-threshold', type=float, help='sigma threshold during marching cube', default=10., show_default=True)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--outdir', type=str, help='output dir of video', required=False, default='out_geometry')
@click.option('--id_name', type=str, required=True)
@click.option('--model_name', type=str, required=True)
@click.option('--original_image', type=str, required=True)
def render(fname, size, sigma_threshold, w_frames, outdir, id_name, model_name, original_image):
    os.makedirs(outdir, exist_ok=True)
    
    sigma_threshold = sigma_threshold
    num = w_frames
    voxel_grid = np.load(fname)
    voxel_grid = np.maximum(voxel_grid, 0)
    vertices, triangles = mcubes.marching_cubes(voxel_grid, sigma_threshold)
    mesh = trimesh.Trimesh(vertices/size, triangles)

    # mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # mesh.fix_normals()
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True, material=pyrender.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        smooth=True,
        metallicFactor=0.2,
        roughnessFactor=0.6,
    ))

    video_writer = imageio.get_writer(f"{outdir}/interpolation_{id_name}_render_{model_name}.mp4", mode='I', fps=60, codec='libx264', bitrate='10M')
    images_out_interpolate = []
    for i in tqdm(range(num)):

        pitch_range = 0.25
        yaw_range = 0.35
        frame_idx = i
        num_keyframes = 1
        camera_lookat_point = torch.tensor([0, 0, 0])
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(-2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                3.14/2 -0.05 + pitch_range * np.cos(-2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                                                camera_lookat_point, radius=2.7)
        intrinsic_matrix = np.array([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]]) # pixel unit
        pixel_const = 512 # eg3d is normalizing intrinsics by 512 (image size)
        camera = pyrender.IntrinsicsCamera(fx=4.2647 * pixel_const, fy=4.2647 * pixel_const, cx=0.5 * pixel_const, cy=0.5 * pixel_const)
        P = cam2world_pose.squeeze().reshape(4,4).numpy()
        # this is cam2world in opencv coordinate, we need world2cam in opengl coordinate
        P = np.linalg.inv(P)
        P[:3, :3] = P[:3, :3].T
        P[:3, 3] = -P[:3, :3] @ P[:3, 3]
        P = P @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flip y and z axis
        P[:3, 3] += 0.5 # move to center of the image


        yaw = math.pi*(0.5+0.15*math.cos(2*math.pi*i/num))
        pitch = math.pi*(0.5-0.05*math.sin(2*math.pi*i/num))
        scene = pyrender.Scene(ambient_light=[5, 5, 5], bg_color=[255, 255, 255])
        # camera = pyrender.PerspectiveCamera(yfov=np.pi / 180.0 * 18)
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
        scene.add(mesh, pose=np.eye(4))
        # camera_points, phi, theta = sample_camera_positions(device=None, n=1, r=2.7, horizontal_mean=yaw, vertical_mean=pitch, mode=None)
        # c = create_cam2world_matrix(-camera_points, camera_points, device=None)
        # P = c.reshape(-1,4,4).numpy()[0]
        # P[:3, 3] += 0.5

        scene.add(camera, pose=P)
        scene.add(direc_l, pose=P)

        # render scene
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)

        im = Image.fromarray(color)
        # flip image left-right
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
        images_out_interpolate.append(im)

        r.delete()
        video_writer.append_data(np.asarray(im).astype(np.uint8))
    video_writer.close()
    # save interpolation video
    n = 5
    indices = np.linspace(0, len(images_out_interpolate)-1, n).astype(int)[1:-1]
    images_out = [images_out_interpolate[i] for i in indices]

    # put input image at the beginning
    input_img = PIL.Image.open(original_image).convert('RGB')
    input_img = np.array(input_img) # (512, 512, 3)
    images_out = [input_img] + images_out

    # stack horizontally
    images_out = np.hstack(images_out)
    # # save
    # PIL.Image.fromarray(images_out).save(os.path.join(outdir, 'interpolate_video_{}_{}_geometry.pdf'.format(id_name, model_name)))
    # save png
    PIL.Image.fromarray(images_out).save(os.path.join(outdir, 'interpolate_video_{}_{}_geometry.png'.format(id_name, model_name)))
    
    # img2video(sorted(glob.glob(f"tmp/{id}/*.png")), f"{outdir}/render.mp4")
    # cmd = f'rmdir /s/q tmp\\{id}'
    # os.system(cmd)
    

def img2video(img_list, mp4):
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', bitrate='10M')
    for img in img_list:
        frame = cv2.imread(img)
        video_out.append_data(frame)
    video_out.close()


if __name__ == '__main__':
    # fname = 'out/1.npy'
    render()

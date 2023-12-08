import argparse
import os
import shutil
import numpy as np
from itertools import combinations
from camera_utils import LookAtPoseSampler
from PIL import Image
from utils_copy import id_utils # make a copy of utils to avoid import error as eg3d also has utils
import torch
import json
import dnnlib
import legacy
from torchvision.utils import make_grid, save_image
from torchvision.transforms import CenterCrop
import imageio
from utils.reconstruction_utils import networks_dir

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

def format_data(t):
    folder = '/playpen-nas-ssd/awang/data/luchao_preprocessed_subset_t{}'.format(t)
    ids = []
    image_paths = os.listdir(folder)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('_')[0]
            if id_name not in ids:
                ids.append(id_name)
    ids.sort()
    id_name = ids[0]

    ws = []
    input_features = []
    for fileName in image_paths:
        if fileName.endswith('_latent.npy') and 'mirror' not in fileName:
            if (fileName.split('_')[0] == id_name):
                name = fileName[:-11]
                ws.append((name, np.load(os.path.join(folder, fileName), allow_pickle=True)))
            
        if fileName.endswith('.png') and 'mirror' not in fileName:
            if (fileName.split('_')[0] == id_name):
                img = Image.open(os.path.join(folder, fileName)).convert('RGB')
                img = np.array(img)
                feature = person_identifier.get_feature(img)
                input_features.append(feature)
    return id_name, ws, input_features

device=torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="which GPU to use")
parser.add_argument("-t", "--time", help="time")
parser.add_argument('-b', '--bound') # either upper or lower
parser.add_argument('-l', '--lora',
                    action='store_true') # on/off flag 

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# get correct model
if args.bound == 'upper':
    if args.lora:
        model_name = 'upper_bound_lora'
    else:
        model_name = 'upper_bound_no_lora'
else:
    if args.lora:
        model_name = 'lower_bound_lora'
    else:
        model_name = 'lower_bound_no_lora'    
network_pkl = networks_dir + model_name + '.pkl'

t = (int)(args.time)

result = 0
pitch_range = 0.25
yaw_range = 0.35
num_rotations = 60
num_ticks = 10
id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)
transform = CenterCrop([528, 5152])

#for t in range(5):
with torch.no_grad():
    # get all training samples and their corresponding ws
    _, ws, input_features = format_data(t)
    out_dir_root = 'out/' + model_name + '_interpolations'
    if not (os.path.isdir(out_dir_root)):
        os.mkdir(out_dir_root)
    out_dir = os.path.join(out_dir_root, f't{t}')
    if not (os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    else:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.mkdir(out_dir)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to(device) # type: ignore

    pairs = combinations(ws, 2)
    final_sim_list = []
    for k, pair in enumerate(pairs):
        print(f'interpolating pair {k}')
        w1, w2, name1, name2 = pair[0][1], pair[1][1], pair[0][0], pair[1][0]

        sim_list = [0 for _ in range(num_ticks)]
        video_out = imageio.get_writer(f'{out_dir}/{k}--{name1}--{name2}.mp4', mode='I', fps=60, codec='libx264')
        for idx in range(num_rotations):
            camera_lookat_point = torch.tensor([0, 0, 0], device=device)
            cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * idx / num_rotations),
                                                    3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * idx / num_rotations),
                                                    camera_lookat_point, radius=2.7, device=device)
            intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
            c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).type(torch.float32) # (1, 25)
            imgs = []
            
            for i, alpha in enumerate(np.linspace(0, 1, num_ticks)):
                w = alpha * w1 + (1 - alpha) * w2
                img = G.synthesis(ws=torch.from_numpy(w).to(device), c=c[0:1].to(device), noise_mode='const')['image'][0]
                img = (img * 127.5 + 128).clamp(0, 255).cpu()
                permuted = np.array(img.permute(1, 2, 0).to(torch.uint8).cpu())
                feature = person_identifier.get_feature(permuted)
                sim = max([person_identifier.compute_similarity(feature, v) for v in input_features]).item()
                sim_list[i] += sim
                
                imgs.append(img)

            grid = transform(make_grid(imgs, nrow=10, normalize=True))
            #grid = (grid * 127.5 + 128).clamp(0, 255).cpu()
            #grid = np.array(grid.permute(1, 2, 0).to(torch.uint8).cpu())
            grid = grid.permute(1, 2, 0).cpu().numpy()
            
            video_out.append_data(grid)

        sim_list = [s / num_rotations for s in sim_list]
        sim_mean = round(np.mean(sim_list), 3)
        final_sim_list.append(sim_mean)
        video_out.close()

    result = np.mean(final_sim_list)
    print(f'mean ID similarity for time {t} is {result}')
    
interpolation_location = 'out/interpolation_output.json'
if not os.path.isfile(interpolation_location):
    output = {'upper_bound_no_lora': [0 for _ in range(5)],
              'upper_bound_lora': [0 for _ in range(5)],
              'lower_bound_no_lora': [0 for _ in range(5)],
              'lower_bound_lora': [0 for _ in range(5)]}
else:
    with open(interpolation_location) as feedsjson:
        output = json.load(feedsjson)

output[model_name][t] = result
with open(interpolation_location, 'w') as f:
    json.dump(output, f)

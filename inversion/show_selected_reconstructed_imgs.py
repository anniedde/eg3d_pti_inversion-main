import os
import argparse
import matplotlib.pyplot as plt
import PIL
import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path, id_model, \
                                        networks_dir, embeddings_folder, evaluate_metrics
import json
import numpy as np
from camera_utils import LookAtPoseSampler
from PIL import Image
import dnnlib
import legacy
from torchvision.transforms import CenterCrop
import imageio

device=torch.device('cuda')
parser = argparse.ArgumentParser()
parser.add_argument('--pti',
                    action='store_true') # on/off flag 
parser.add_argument('--gpu', required=False)
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models_list = ['upper_bound_no_lora', 'upper_bound_lora', 'lower_bound_no_lora_alternate', 'lower_bound_lora_alternate']
transform = transforms.Compose([
                    transforms.ToTensor()
                ])

out_dir = f'out'
selected_imgs = [(0, '2023-02-25-23-55-08_017'),
                 (1, '2023-03-16-02-27-06_015'),
                 (2, '2023-03-27-02-04-37_017'),
                 (3, '2023-04-12-22-15-42_028')]#,
                 #(4, '2023-06-24-17-14-18_025')]

pitch_range = 0.25
yaw_range = 0.35
num_rotations = 120
if args.pti:
    video_out = imageio.get_writer(f'{out_dir}/selected_reconstructions_pti_alternate.mp4', mode='I', fps=60, codec='libx264')
else :
    video_out = imageio.get_writer(f'{out_dir}/selected_reconstructions_alternate.mp4', mode='I', fps=60, codec='libx264')
with torch.no_grad():
    for idx in range(num_rotations):
        camera_lookat_point = torch.tensor([0, 0, 0], device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * idx / num_rotations),
                                                3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * idx / num_rotations),
                                                camera_lookat_point, radius=2.7, device=device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).type(torch.float32) # (1, 25)
        imgs = []
        for (t, id_name) in selected_imgs:
            input_loc = os.path.join('embeddings', 'upper_bound_no_lora', f't{t}', id_name, 'input.png')
            input_image = transform(PIL.Image.open(input_loc).convert('RGB'))
            input_image = (input_image * 255).clamp(0, 255).cpu()
            imgs.append(input_image)
            for model_name in models_list:
                network_pkl = f'networks/{model_name}.pkl'
                with dnnlib.util.open_url(network_pkl) as f:
                    G = legacy.load_network_pkl(f)['G_ema'].eval().to(device) # type: ignore
                if args.pti:
                    G.load_state_dict(torch.load(os.path.join('embeddings', model_name, f't{t}',id_name, 'tuned_G.pt')))
                w = np.load(os.path.join('embeddings', model_name, f't{t}',id_name, 'optimized_noise_dict.pickle'), allow_pickle=True)
                w = w['projected_w']
                img = G.synthesis(ws=torch.from_numpy(w).to(device), c=c[0:1].to(device), noise_mode='const')['image'][0]
                img = (img * 127.5 + 128).clamp(0, 255).cpu()
                imgs.append(img)

        grid = make_grid(imgs, nrow=5, normalize=True)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        video_out.append_data(grid)
    video_out.close()

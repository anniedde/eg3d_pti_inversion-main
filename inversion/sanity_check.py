import sys
sys.path.append("../")

from torch.utils.tensorboard import SummaryWriter
import torch
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config, hyperparameters
import shutil
import dnnlib
import legacy
import numpy as np
import PIL.Image
import json

from pti_training.coaches.single_id_coach import SingleIDCoach
from pti_training.coaches.single_id_coach_grayscale import SingleIDCoachGrayscale
from utils.ImagesDataset import ImagesDataset, GrayscaleImagesDataset, DECADataset

outdir = 'out'
network_pkl =  '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils/trained_luchao_50_images_no_lora.pkl'
input_pose_path = '/playpen-nas-ssd/awang/data/luchao_one_more_training_image/epoch_20_000000/cameras.json'
# The image tag to lookup in the cameras json file
input_id = '2023-03-27-02-04-37_002'
# Where the input image resides
input_data_path = '/playpen-nas-ssd/awang/data/luchao_one_more_training_image/crop_1024'

os.makedirs(outdir, exist_ok=True)

print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

f = open(input_pose_path)
target_pose = np.asarray(json.load(f)[input_id]['pose']).astype(np.float32)
f.close()
o = target_pose[0:3, 3]
print("norm of origin before normalization:", np.linalg.norm(o))
o = 2.7 * o / np.linalg.norm(o)
target_pose[0:3, 3] = o
target_pose = np.reshape(target_pose, -1)  
intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
target_pose = np.concatenate([target_pose, intrinsics])
target_pose = torch.tensor(target_pose, device=device).unsqueeze(0)

ws = np.load('/playpen-nas-ssd/awang/data/luchao_preprocessed/2023-03-27-02-04-37_002_latent.npy')[0][0]
ws = ws.repeat(1, 0)
ws = torch.tensor(ws, dtype=torch.float32, device=device,
                         requires_grad=True)
ws = (ws).repeat([1, G.backbone.mapping.num_ws, 1])
#camera_params = torch.tensor(np.load('/playpen-nas-ssd/awang/data/luchao_preprocessed/2023-03-27-02-04-37_002.npy').reshape(1,25), device=device)
img = G.synthesis(ws, target_pose, noise_mode='const')['image'].detach().cpu()

img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/trainImage2.png')
import os, sys
import argparse

import torch
from tqdm import tqdm
from lpips import LPIPS
from DISTS_pytorch import DISTS
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
import pickle
import json

sys.path.append('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/utils_copy')
from id_utils import PersonIdentifier

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

def evaluate_metrics(folder, lpips, dists, person_identifier):
    # check that metrics.json does not exist
    if os.path.exists(os.path.join(folder, 'metrics.json')):
        return

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    lpips_loss_list, psnr_list, dists_loss_list, id_error_list = [], [], [], []
    output_img = Image.open(os.path.join(folder, 'before_pti_rgb_proj.png')).convert('RGB') # [512, 512, 3] [H, W, C]
    input_img = Image.open(os.path.join(folder, 'input.png')).convert('RGB') # [512, 512, 3] [H, W, C]
    
    input_img = transform(input_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
    output_img = transform(output_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
    
    # ----------------------------------- LPIPS ----------------------------------- #
    # lpips requires normalized images [-1, 1]
    lpips_loss = lpips(input_img, output_img).squeeze().item()
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
    # DISTS requires normalized images between [0, 1] rather than [-1, 1]
    input_dists, output_dists = (input_img + 1) / 2.0, (output_img + 1) / 2.0
    dists_loss = dists(input_dists, output_dists)
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
    
    # save metrics to json
    metrics = {
        'lpips': lpips_loss,
        'psnr': psnr,
        'dists': dists_loss,
        'id_sim': sim
    }
    
    with open(os.path.join(folder, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

def eval_reconstruction_results(celeb, experiment, time_model, lpips, dists, person_identifier):
    
    root_folder = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment)
    for eval_t in range(time_model + 1):
        results_folder = os.path.join(root_folder, str(time_model), str(eval_t))
        print('Evaluating reconstruction for celeb: {}, experiment: {}, time: {}'.format(celeb, experiment, eval_t))
        if not os.path.exists(results_folder):
            print('Folder does not exist: {}'.format(results_folder))
            continue
        for image in os.listdir(results_folder):
            evaluate_metrics(os.path.join(results_folder, image), lpips, dists, person_identifier)

def eval_reconstruction_results_upper_bound(celeb, experiment, lpips, dists, person_identifier):
    
    root_folder = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment)
    for eval_t in range(10):
        results_folder = os.path.join(root_folder, 'all', str(eval_t))
        print('Evaluating reconstruction for celeb: {}, experiment: {}, time: {}'.format(celeb, experiment, eval_t))
        if not os.path.exists(results_folder):
            print('Folder does not exist: {}'.format(results_folder))
            continue
        for image in os.listdir(results_folder):
            evaluate_metrics(os.path.join(results_folder, image), lpips, dists, person_identifier)

def process_args():
    parser = argparse.ArgumentParser(description='Batch Train Celeb')
    
    # Required arguments
    parser.add_argument('--celeb', type=str, help='Name of the celebrity', required=True)
    parser.add_argument('--experiment', type=str, help='Name of the experiment', required=True)

    # Optional arguments
    parser.add_argument('--device', type=str, help='GPUs to use', required=False, default='0')
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    return args

if __name__ == '__main__':
    args = process_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS(net='alex').eval().to(device)
    dists = DISTS().to(device)
    person_identifier = PersonIdentifier('/playpen-nas-ssd/awang/mystyle_original/third_party/model_ir_se50.pth', None, None)
    if args.experiment == 'upper_bound':
        eval_reconstruction_results_upper_bound(args.celeb, args.experiment, lpips, dists, person_identifier)
    else:
        for t in range(20):
            eval_reconstruction_results(args.celeb, args.experiment, t, lpips, dists, person_identifier)

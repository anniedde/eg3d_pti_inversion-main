import numpy as np
import json
import torch
import os
import pickle
import PIL
from lpips import LPIPS
from DISTS_pytorch import DISTS
from torchvision import transforms
from configs import paths_config
from utils_copy import id_utils
import math

paths_config_path = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/configs/paths_config.py'
id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
networks_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/orange_cat/'
embeddings_folder = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/luchao/'

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

def evaluate_metrics(folder, pti, lpips, dists, person_identifier, device=torch.device('cuda')):
    lpips_loss_list, psnr_list, dists_loss_list, id_error_list = [], [], [], []
    
    for id_name in os.listdir(folder):
        if os.path.isdir(os.path.join(folder, id_name)) and os.path.exists(os.path.join(folder, id_name, 'final_rgb_proj.png')):
            if pti:
                output_img = PIL.Image.open(os.path.join(folder, id_name, 'final_rgb_proj.png')).convert('RGB') # [512, 512, 3] [H, W, C]
            else:
                output_img = PIL.Image.open(os.path.join(folder, id_name, 'before_pti_rgb_proj.png')).convert('RGB') # [512, 512, 3] [H, W, C]
            input_img = PIL.Image.open(os.path.join(folder, id_name, 'input.png')).convert('RGB') # [512, 512, 3] [H, W, C]
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_img = transform(input_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
            output_img = transform(output_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
            # lpips requires normalized images [-1, 1]
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
            id_error = 1 - sim

            lpips_loss_list.append(lpips_loss)
            psnr_list.append(psnr)
            dists_loss_list.append(dists_loss)
            id_error_list.append(id_error)
            
    lpips_mean = sum(lpips_loss_list) / len(lpips_loss_list)
    psnr_mean = sum(psnr_list) / len(psnr_list)
    dists_mean = sum(dists_loss_list) / len(dists_loss_list)
    id_error_mean = sum(id_error_list) / len(id_error_list)

    return lpips_mean, psnr_mean, dists_mean, id_error_mean

def get_id_error(image_path, reference_folder, person_identifier, device=torch.device('cuda')):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img = PIL.Image.open(image_path).convert('RGB') # [512, 512, 3] [H, W, C]
    img = transform(img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
    img_denorm = denorm(img)                       # id_error requires denormalized images
    img_feature = person_identifier.get_feature(img_denorm.squeeze())

    lowest_error = math.inf
    for item in os.listdir(reference_folder):
        if item.endswith('.png'):
            reference_path = os.path.join(reference_folder, item)

            reference_img = PIL.Image.open(reference_path).convert('RGB')
            reference_img = transform(reference_img).to(device).unsqueeze(0)
            reference_img_denorm = denorm(reference_img)
            reference_img_feature = person_identifier.get_feature(reference_img_denorm.squeeze())
            
            sim = person_identifier.compute_similarity(img_feature, reference_img_feature)
            sim = sim.item()
            sim = round(sim, 3)
            id_error = 1 - sim

            if id_error < lowest_error:
                lowest_error = id_error

    return lowest_error
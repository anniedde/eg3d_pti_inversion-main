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

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--t')
parser.add_argument('--pti',
                    action='store_true') # on/off flag 
parser.add_argument('--gpu', required=False)
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models_list = ['lower_bound', 't0_upper_bound', 't0_t1_upper_bound', 'rank_expansion', 'replay_regular_lora_expansion', 'lora_svd_rank1_1step', 'replay_lora_svd_rank1']
out_dir = 'out/reconstructions'

transform = transforms.Compose([
                    transforms.ToTensor()
                ])
t = args.t
folder = '/playpen-nas-ssd/awang/data/Margot_1step/0/test/preprocessed'
dataset_json = os.path.join(folder, 'cameras.json')
image_paths = os.listdir(folder)

if args.pti:
    out_dir = f'out/reconstructions_pti_alternate/t{t}'
else:
    out_dir = f'out/reconstructions_no_pti_alternate/t{t}'
entire_time_list = [None for _ in range(50)]
img_count = 0
for fileName in image_paths:
    if fileName.endswith('.png') and 'mirror' not in fileName:
        id_name = fileName.split('.')[0]
        image_folder = os.path.join(folder, id_name)
        input_loc = os.path.join(image_folder, fileName)

        input_image = transform(PIL.Image.open(input_loc).convert('RGB'))
        imgs = [input_image]
        entire_time_list[img_count] = input_image
        for i, model_name in enumerate(models_list):
            if args.pti:
                img_loc = os.path.join('embeddings', 'Margot_1step', model_name, f'{t}', id_name, 'final_rgb_proj.png')
            else:
                img_loc = os.path.join('embeddings', 'Margot_1step', model_name, f'{t}', id_name, 'before_pti_rgb_proj.png')
            img = transform(PIL.Image.open(img_loc).convert('RGB'))
            imgs.append(img)
            print(f'i: {i}, img_count: {img_count}, (i + 1) * 10 + img_count: {(i + 1) * 10 + img_count}')
            #entire_time_list[(i + 1) * 2 + img_count] = img
        grid = make_grid(imgs, normalize=True)

        # save grid
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        out_loc = os.path.join(out_dir, fileName)
        save_image(grid, out_loc)
        img_count += 1

#entire_time_grid = make_grid(entire_time_list, nrow=5, normalize=True)
#save_image(entire_time_grid, os.path.join(out_dir, 'grid.png'))

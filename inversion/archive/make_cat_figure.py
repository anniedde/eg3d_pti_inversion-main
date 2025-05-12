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

parser.add_argument('--pti',
                    action='store_true') # on/off flag 
parser.add_argument('--gpu', required=False)
args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
models_list = ['no_lora', 'lora']
out_dir = 'out/reconstructions'

transform = transforms.Compose([
                    transforms.ToTensor()
                ])

folder = '/playpen-nas-ssd/awang/data/luchao_test_minus_subset'
dataset_json = os.path.join(folder, 'epoch_20_000000', 'cameras.json')
image_paths = os.listdir(folder)

if args.pti:
    out_dir = f'out/reconstructions_pti_alternate/t{t}'
else:
    out_dir = f'out/reconstructions_no_pti_alternate/t{t}'
entire_time_list = [None for _ in range(50)]

imgs = ['6890', '7718', '7720', '9909']

for img_name in imgs:
    personalized_embedding_path = f'embeddings/orange_cat_original/personalized/IMG_6890'
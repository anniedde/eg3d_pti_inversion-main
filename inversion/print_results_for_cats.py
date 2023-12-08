import os
import argparse
import matplotlib.pyplot as plt
import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path, id_model, \
                                        networks_dir, embeddings_folder, evaluate_metrics
import json


parser = argparse.ArgumentParser()
parser.add_argument('--pti',
                    action='store_true') # on/off flag 
args = parser.parse_args()

models_list = ['lora', 'no_lora']
lpips_map, psnr_map, dists_map, id_error_map = {}, {}, {}, {}

device=torch.device('cuda')
lpips = LPIPS(net='alex').to(device).eval()
dists = DISTS().to(device)
person_identifier = id_utils.PersonIdentifier(id_model, None, None)
for model_name in models_list:

    embedding_folder = os.path.join('embeddings', 'luchao_subset_test_imgs', model_name)
    lpips_loss, psnr, dists_loss, id_error = evaluate_metrics(embedding_folder, args.pti, lpips, dists, person_identifier)
    
    lpips_map[model_name] = lpips_loss
    psnr_map[model_name] = psnr
    dists_map[model_name] = dists_loss
    id_error_map[model_name] = id_error

ret = {}
ret['lpips'] = lpips_map
ret['psnr'] = psnr_map
ret['dists'] = dists_map
ret['id_error'] = id_error_map

if args.pti:
    out_loc = 'out/luchao_subset/reconstruction_results_test_imgs_pti.json'
else:
    out_loc = 'out/luchao_subset/reconstruction_results_test_imgs_no_pti.json'
with open(out_loc, 'w') as out:
    json.dump(ret, out)
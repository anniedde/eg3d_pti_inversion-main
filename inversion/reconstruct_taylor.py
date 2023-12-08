# for a specific trained network, reconstruct all the test images first without PTI and then with.
import argparse
import os
import torch
#from utils_copy import id_utils
#from utils.reconstruction_utils import paths_config_path
#import legacy
import torch
from multiprocessing import Pool
#import glob
from run_pti import run_PTI_func

import os

def run_pti(model_name, year):
    data_folder = os.path.join(f'/playpen-nas-ssd/awang/data/Taylor/Taylor_Swift_{year}', 'test', 'preprocessed')
    print('Running inversion for model: {}'.format(model_name) + ' on data: {}'.format(data_folder))
    network_pkl = os.path.join('networks', 'Taylor', model_name + '.pkl')
    dataset_json = os.path.join(data_folder, 'cameras.json')
    image_paths = os.listdir(data_folder)
    embedding_folder = os.path.join('Taylor', model_name, year)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(data_folder, id_name)
            
            if not os.path.exists(os.path.join('embeddings', embedding_folder, id_name)):

                # run the script
                print('running pti for img={}'.format(id_name))
                run_PTI_func(id_name, dataset_json, image_folder, embedding_folder, network_pkl)

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--upper',
                    action='store_true') # on/off flag 
parser.add_argument('--year', type=str, required=True)
parser.add_argument('--gpu', required=False)

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# get correct model
if args.upper:
    model_name = 'upper_bound'
else:
    model_name = 'lower_bound'

device=torch.device('cuda')


run_pti(model_name, args.year)

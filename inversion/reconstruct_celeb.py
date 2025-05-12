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

def run_pti(experiment, model_vid, year, celeb):
    data_folder = os.path.join(f'/playpen-nas-ssd/awang/data/eg3d_replay/{celeb}/{year}', 'test', 'preprocessed')
    print('Running inversion for experiment {} model: {}'.format(experiment, model_vid) + ' on data: {}'.format(data_folder))
    network_pkl = os.path.join('networks', celeb, experiment, model_vid + '.pkl')
    dataset_json = os.path.join(data_folder, 'cameras.json')
    image_paths = os.listdir(data_folder)
    embedding_folder = os.path.join(celeb, experiment, model_vid, year)
    for fileName in image_paths:
        if fileName.endswith('.png') and 'mirror' not in fileName:
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(data_folder, id_name)
            
            if not os.path.exists(os.path.join('embeddings', embedding_folder, id_name, 'before_pti_rgb_proj.png')):

                # run the script
                print('running pti for img={}'.format(id_name))
                try:
                    run_PTI_func(id_name, dataset_json, image_folder, embedding_folder, network_pkl)
                except Exception as e:
                    print('error: ', e)
                    exit()

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, required=True)
parser.add_argument('--celeb', type=str, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--gpu', required=False)

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# get correct model
experiment = args.experiment
model_vid = args.model
device=torch.device('cuda')


run_pti(experiment, model_vid, args.year, args.celeb)

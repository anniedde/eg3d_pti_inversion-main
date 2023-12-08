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

def run_pti():
    data_folder = '/playpen-nas-ssd/awang/data/barack/Barack_test_easy'
    print('Running inversion on data: {}'.format(data_folder))
    network_pkl = '/playpen-nas-ssd/awang/my3dgen/eg3d/training-runs/00102-ffhq-Barack_20/network-snapshot-001400.pkl'
    dataset_json = os.path.join(data_folder, 'epoch_20_000000', 'cameras.json')
    image_paths = os.listdir(data_folder)
    embedding_folder = os.path.join('Barack_trained_on_20')
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(data_folder, id_name)
            
            if not os.path.exists(os.path.join('embeddings', embedding_folder, id_name)):

                # run the script
                print('running pti for img={}'.format(id_name))
                run_PTI_func(id_name, dataset_json, image_folder, embedding_folder, network_pkl)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', required=False)

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

run_pti()

# for a specific trained network, reconstruct all the test images first without PTI and then with.
import argparse
import os
import shutil
import torch
import dnnlib
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path
import legacy

def run_pti(model_name, network_pkl):
    folder = '/playpen-nas-ssd/awang/data/Barack_test_easy'
    dataset_json = os.path.join(folder, 'epoch_20_000000', 'cameras.json')
    image_paths = os.listdir(folder)
    embedding_folder = os.path.join('Barack_easy', model_name)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(folder, id_name)
            
            #if not os.path.exists(os.path.join('embeddings', embedding_folder, id_name)):
            if True:
                # update paths_config
                os.system('sed -i "s/input_id = .*/input_id = \'{}\'/g" {}'.format(id_name, paths_config_path))
                os.system('sed -i "s@input_pose_path = .*@input_pose_path = \'{}\'@g" {}'.format(dataset_json, paths_config_path))
                os.system('sed -i "s@input_data_path = .*@input_data_path = \'{}\'@g" {}'.format(image_folder, paths_config_path))
                os.system('sed -i "s@embedding_folder = .*@embedding_folder = \'{}\'@g" {}'.format(embedding_folder, paths_config_path))
                os.system('sed -i "s@eg3d_ffhq = .*@eg3d_ffhq = \'{}\'@g" {}'.format(network_pkl, paths_config_path))
                
                # run the script
                print('running pti for img={}'.format(id_name))
                cmd = 'python run_pti.py'
                os.system(cmd)

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lora',
                    action='store_true') # on/off flag 
parser.add_argument('--gpu', required=False)

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# get correct model
if args.lora:
    model_name = 'Barack_lora'
    network_pkl = 'networks/Barack_lora.pkl'
else:
    model_name = 'Barack_no_lora'
    network_pkl = 'networks/Barack_no_lora.pkl'


device=torch.device('cuda')

run_pti(model_name, network_pkl)
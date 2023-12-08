# for a specific trained network, reconstruct all the test images first without PTI and then with.
import argparse
import os
import shutil
import torch
import dnnlib
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path, id_model, \
                                        embeddings_folder, evaluate_metrics
import legacy
from run_pti import run_PTI_func

def run_pti(model_name, network_pkl):
    folder = '/playpen-nas-ssd/awang/data/orange_cat_test'
    dataset_json = os.path.join(folder, 'epoch_20_000000', 'cameras.json')
    image_paths = os.listdir(folder)
    embedding_folder = os.path.join('orange_cat', model_name)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('.')[0]
            image_folder = os.path.join(folder, id_name)
            
            if not os.path.exists(os.path.join('embeddings', embedding_folder, id_name)):
            #if True:

                # run the script
                print('running pti for img={}'.format(id_name))
                run_PTI_func(id_name, dataset_json, image_folder, embedding_folder, network_pkl)


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pretrained',
                    action='store_true') # on/off flag 
parser.add_argument('--gpu', required=False)

args = parser.parse_args()

if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# get correct model
networks_dir = 'networks/orange_cat/'
if args.pretrained:
    model_name = 'pretrained'
else:
    model_name = 'upper_bound_no_lora'
network_pkl = networks_dir + model_name + '.pkl'

device=torch.device('cuda')

run_pti(model_name, network_pkl)

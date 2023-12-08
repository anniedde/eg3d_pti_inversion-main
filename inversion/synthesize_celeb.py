import os
import json
#from utils.reconstruction_utils import get_id_error
import math
import argparse
from utils_copy import id_utils
import PIL
import torch
from utils.reconstruction_utils import get_id_error
import torchvision.transforms as transforms
import requests
from tqdm import tqdm

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message


id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)

def run_generate_script(celeb, t, model_name):
    command = 'python generate.py ' \
            + f'--anchors_path=/playpen-nas-ssd/awang/data/{celeb}/{t}/train/preprocessed ' \
            + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{model_name}.pkl ' \
            + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{t}/{model_name} ' 
    os.system(command)

def evaluate(celeb, year, model):
    eval_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{year}/{model}/'

    mean_id_error = 0
    ret = {}

    for im in tqdm(os.listdir(os.path.join(eval_dir, 'images'))):
        im_path = os.path.join(eval_dir, 'images', im)

        # get lowest distance from reference set
        lowest_dist = math.inf
        reference_folder = f'/playpen-nas-ssd/awang/data/{celeb}/all/all'
        dist = get_id_error(im_path, reference_folder, person_identifier)
        if dist < lowest_dist:
            lowest_dist = dist
        
        ret[im] = lowest_dist
        mean_id_error += lowest_dist

    mean_id_error /= len(os.listdir(os.path.join(eval_dir, 'images')))
    ret['mean_id_error'] = mean_id_error
    print('mean dist: ', mean_id_error)

    # Write mean_id_error to a JSON file
    json_file_path = os.path.join(eval_dir, 'mean_id_error.json')
    with open(json_file_path, 'w') as f:
        json.dump(ret, f)

    notify(f'Finished evaluating synthesis results for {celeb} {model}, t={year}.')

def evaluate_across_years(celeb):
    for model in ['upper_bound', 'lower_bound']:
        eval_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/all/{model}/'

        mean_id_error = 0
        ret = {}

        for im in tqdm(os.listdir(os.path.join(eval_dir, 'images'))):
            im_path = os.path.join(eval_dir, 'images', im)

            # get lowest distance from reference set
            lowest_dist = math.inf
            reference_folder = f'/playpen-nas-ssd/awang/data/{celeb}/all/all'
            dist = get_id_error(im_path, reference_folder, person_identifier)
            if dist < lowest_dist:
                lowest_dist = dist
            
            ret[im] = lowest_dist
            mean_id_error += lowest_dist
            
            ret[im] = lowest_dist
            mean_id_error += lowest_dist

        mean_id_error /= len(os.listdir(os.path.join(eval_dir, 'images')))
        ret['mean_id_error'] = mean_id_error
        print('mean dist: ', mean_id_error)

        # Write mean_id_error to a JSON file
        json_file_path = os.path.join(eval_dir, 'mean_id_error.json')
        with open(json_file_path, 'w') as f:
            json.dump(ret, f)

        notify(f'Finished evaluating synthesis results for {celeb} {model} (all years).')

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--celeb', type=str, required=True)
    parser.add_argument('--gpu', required=False, default='0')

    args = parser.parse_args()
    celeb = args.celeb

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t = args.year

    #os.chdir("../../semantic_editing/mystyle")
    #run_generate_script(args.celeb, args.year, 'lower_bound')
    #run_generate_script(args.celeb, args.year, 'upper_bound')
    #os.chdir("../../eg3d_pti_inversion-main/inversion")



    for model in ['lower_bound', 'upper_bound']:
        evaluate(celeb, args.year, model)
        
        

    notify(f'Finished evaluating all synthesis results for {celeb}')

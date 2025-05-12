import os
import multiprocessing
import argparse
import requests
from synthesize_celeb import evaluate_across_years, evaluate
import traceback
import time
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
import glob
import pickle
import subprocess
import json
from cleanfid import fid
from tqdm import tqdm
import torchvision.transforms as transforms

os.environ['MKL_NUM_THREADS'] = '1'

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def run_generate_script_by_time(celeb, t, experiment, model_name, gpu='0'):
    anchor_celeb = celeb #if 'Margot' not in celeb else 'Margot'
    synthesized_folder = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{experiment}/{model_name}/{t}/images'
    if os.path.exists(synthesized_folder) and len(os.listdir(synthesized_folder)) >= 20:
        print(f'Synthesized images already exist for {celeb} {experiment} {model_name} {t}')
        return

    command = 'conda run -n mystyle_new python generate.py ' \
            + f'--anchors_path=/playpen-nas-ssd/awang/data/eg3d_replay/{anchor_celeb}/{t}/train/preprocessed ' \
            + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{experiment}/{model_name}.pkl ' \
            + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{experiment}/{model_name}/{t} ' \
            + f'--device={gpu}'
    print(command)
    os.system(command)

def run_generate_script_all(celeb, models, gpu='0'):
    for model_name in models:
        command = 'conda run -n mystyle_new python generate.py ' \
                + f'--anchors_path=/playpen-nas-ssd/awang/data/{celeb}/all/train/preprocessed/ ' \
                + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{model_name}.pkl ' \
                + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/all/{model_name} ' \
                + f'--device={gpu}'
        os.system(command)

def run_subprocess(celeb, year, experiment, models, gpu):
    subprocess_cmd = f'CUDA_VISIBLE_DEVICES={gpu} python synthesize_celeb.py --celeb={celeb} --year={year} --experiment={experiment} --models={models}'
    print(subprocess_cmd)
    os.system(subprocess_cmd)

def compile_results(celeb):
    save_img_name_map = {
        'final_rgb_proj.png': 'pti',
        'before_pti_rgb_proj.png': 'before_pti',
        'input.png': 'input'
    }

    for model in ['upper_bound', 'lower_bound']:
        for img_name in ['final_rgb_proj.png', 'before_pti_rgb_proj.png', 'input.png']:
            grid_of_grids_list = []
            for t in range(10):
                imgs = []
                for img in os.listdir(os.path.join('embeddings', celeb, model, str(t))):
                    if 'mirror' not in img:
                        img = read_image(os.path.join('embeddings', celeb, model, str(t), img, img_name)).to(torch.float)
                        imgs.append(img)
                grid = make_grid(imgs, nrow=10, normalize=True, range=(-1, 1))
                save_img_name = f'{t}_{save_img_name_map[img_name]}.png'
                save_image(grid, os.path.join('embeddings', celeb, model, save_img_name))

                grid_of_grids_list.append(make_grid(imgs[:5], nrow=1, normalize=True, range=(-1, 1)))
            grid_of_grids = make_grid(grid_of_grids_list, nrow=10, normalize=True, range=(-1, 1))
            save_image(grid_of_grids, os.path.join('embeddings', celeb, model, f'grid_{save_img_name_map[img_name]}.png'))


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--celeb", type=str, required=True)
        parser.add_argument('--num_gpus', type=int, default=4)
        parser.add_argument('--which_gpus', type=str, required=False)
        parser.add_argument('--experiments', type=str, required=False, default=None)
        parser.add_argument('--models', type=str, required=False, default='0,1,2,3,4,5,6,7,8,9')
        parser.add_argument('--years', type=str, required=False, default=None)

        args = parser.parse_args()
        celeb = args.celeb
        print('celeb:', celeb)
        models = args.models.split(',')  #[model.strip() for model in args.models.split(',')]

        if args.which_gpus:
            gpus = args.which_gpus.split(',') 
            num_gpus = len(gpus)
        else:
            num_gpus = torch.cuda.device_count()
            gpus = range(num_gpus)

        networks_folder = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}'
        if args.experiments:
            experiments = args.experiments.split(',')
        else:
            experiments = [e for e in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, e))]
        
        celeb_data_dir = f'/playpen-nas-ssd/awang/data/eg3d_replay/{celeb}'
  
        try: 
            
            # first generate all the synthetic images
            start = time.time()
            
            os.chdir("../../semantic_editing/mystyle")

            for experiment in experiments:
                if experiment == 'upper_bound':
                    model = 'all'
                    for year in range(10):
                        run_generate_script_by_time(celeb, year, experiment, model, gpus[0])
                else:
                    for model in models:
                        years = range(0, int(model) + 1)
                        for year in years:
                            run_generate_script_by_time(celeb, year, experiment, model, gpus[0])
                
            #run_generate_script_all(celeb, models, gpu=gpus[0])
            os.chdir("../../eg3d_pti_inversion-main/inversion")
            
            # then get the id error for each synthesized image
            
            print('Evaluating across years')
            #evaluate_across_years(celeb)
            
            for experiment in experiments:
                if experiment == 'upper_bound':
                    model = 'all'
                    for year in range(10):
                        print(f'Running evaluation for {celeb} {model} for year {year} in experiment {experiment}')
                        start_time = time.time()
                        evaluate(celeb, year, experiment, model)
                        end_time = time.time()
                        print(f'Finished evaluating {celeb} {model} for year {year} in {end_time - start_time} seconds.')
                else:
                    for model in models:
                        years = range(0, int(model) + 1)
                        for year in years:
                            #run_subprocess(celeb, year, experiment, model, gpus[0])
                            print(f'Running evaluation for {celeb} {model} for year {year} in experiment {experiment}')
                            start_time = time.time()
                            evaluate(celeb, year, experiment, model)
                            end_time = time.time()
                            print(f'Finished evaluating {celeb} {model} for year {year} in {end_time - start_time} seconds.')
            
        except Exception as e:
            notify(f"Error in synthesis for {celeb}: {e}")
            exit()

        end = time.time()
        execution_time = (end - start) / 60
        notify(f"Batch synthesis for {celeb} complete. Took {execution_time} minutes")
        
    except Exception as e:
        traceback.print_exc()

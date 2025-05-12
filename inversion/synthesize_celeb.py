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
from pytorch_fid import fid_score
import time
from cleanfid import fid
from PIL import Image

os.environ['MKL_NUM_THREADS'] = '1'

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

def open_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).to(device).unsqueeze(0)
    img = denorm(img).squeeze()
    return img

def evaluate_metrics(results_folder, reference_folder, reference_name, person_identifier):
    ### ID error
    mean_id_sim = 0

    reference_features = [person_identifier.get_feature(open_image(os.path.join(reference_folder, im))) for im in os.listdir(reference_folder) if im.endswith('.png')]

    for im in tqdm(os.listdir(os.path.join(results_folder))):
        im_path = os.path.join(results_folder, im)

        # get lowest distance from reference set
        im_feature = person_identifier.get_feature(open_image(im_path))
        sims = [person_identifier.compute_similarity(im_feature, reference_feature).item() for reference_feature in reference_features]
        max_sim = max(sims)
        mean_id_sim += max_sim

    mean_id_sim /= len(os.listdir(os.path.join(results_folder)))
    
    ### FID
    if not fid.test_stats_exists(reference_name, mode='clean'):
        print('test stat does not exist')
        fid.make_custom_stats(reference_name, reference_folder, mode="clean")
    #fid_score = fid.compute_fid(results_folder, reference_folder)
    fid_score = fid.compute_fid(results_folder, dataset_name=reference_name, mode="clean", dataset_split="custom")

    return fid_score, mean_id_sim

def run_generate_script(celeb, t, model_name):
    command = 'python generate.py ' \
            + f'--anchors_path=/playpen-nas-ssd/awang/data/{celeb}/{t}/train/preprocessed ' \
            + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{model_name}.pkl ' \
            + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{t}/{model_name} ' 
    os.system(command)

def evaluate(celeb, year, experiment, model):
    try:
        eval_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{experiment}/{model}/{year}'
        json_file_path = os.path.join(eval_dir, 'metrics.json')
        if os.path.exists(json_file_path):
            print(f'Metrics already evaluated for {celeb} {model} {year}.')
            return
        else:
            reference_dir = f'/playpen-nas-ssd/awang/data/eg3d_replay/{celeb}/{year}/test/crop'
            images_dir = os.path.join(eval_dir, 'images')
            reference_name = f'{celeb.lower()}_{year}_my3dgen'
            fid_value, mean_id_sim = evaluate_metrics(images_dir, reference_dir, reference_name, person_identifier)
            ret = {'fid': fid_value, 'id_sim': mean_id_sim}
            # Write mean_id_error to a JSON file
            with open(json_file_path, 'w') as f:
                json.dump(ret, f)
    except Exception as e:
        print(e)
        notify(f'Error in evaluating synthesis results for {celeb} {model}: {e}')
    
    #notify(f'Finished evaluating synthesis results for {celeb} {model}, t={year}.')

def evaluate_across_years(celeb, models):
    #for model in ['upper_bound', 'lower_bound']:
    synthesized_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/all/'
    #reference_dir = f'/playpen-nas-ssd/awang/data/{celeb}/all/all'

    # REMOVE THIS LATER -- HARDCODED
    reference_dir = '/playpen-nas-ssd/awang/data/Margot_1step/all/all'
    print('HARD CODED REFERENCE DIR TO MARGOT_1STEP!!!')
    print('reference_dir: ', reference_dir)
    print('models: ', models)
    
    for model in models:
        eval_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/all/{model}/'
        images_dir = os.path.join(eval_dir, 'images')
        try:
            fid_value = fid_score.calculate_fid_given_paths([reference_dir, images_dir], batch_size=50, device='cuda', dims=2048)
        except Exception as e:
            print(e)
        print(f'FID for {celeb} {model}: {fid_value}')

        # Write FID score to a JSON file
        ret = {'fid': fid_value}
        json_file_path = os.path.join(eval_dir, 'fid_score.json')
        with open(json_file_path, 'w') as f:
            json.dump(ret, f)
        """
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
        """
        notify(f'Finished evaluating synthesis results for {celeb} {model} (all years).')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--celeb', type=str, required=True)
    parser.add_argument('--gpu', required=False, default='0')
    parser.add_argument('--models', required=True)
    parser.add_argument('--experiment', required=True)

    args = parser.parse_args()
    celeb = args.celeb

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    t = args.year
    os.environ['MKL_NUM_THREADS'] = '1'

    #os.chdir("../../semantic_editing/mystyle")
    #run_generate_script(args.celeb, args.year, 'lower_bound')
    #run_generate_script(args.celebexport MKL_NUM_THREADS=1, args.year, 'upper_bound')
    #os.chdir("../../eg3d_pti_inversion-main/inversion")

    # get models
    synthesized_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{t}/'
    #models = os.listdir(synthesized_dir)
    models = [model.strip() for model in args.models.split(',')]
    #evaluate_across_years(celeb)
    for model in models:
        print(f'Evaluating celeb={celeb}, experiment={args.experiment}, model={model}, for year {args.year}...')
        try:
            start_time = time.time()
            evaluate(celeb, args.year, args.experiment, model)
            end_time = time.time()
            print(f'Finished evaluating {celeb} {model} for year {args.year} in {end_time - start_time} seconds.')
        except Exception as e:
            notify(f'Error in evaluating synthesis results for {celeb} {model}: {e}')
        
        

    notify(f'Finished evaluating all synthesis results for {celeb}')

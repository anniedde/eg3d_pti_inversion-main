import os
import sys
import multiprocessing
import argparse
import requests
import traceback
from utils.reconstruction_utils import get_id_error
from utils_copy import id_utils
import itertools
from random import shuffle
import json
from torchvision.utils import save_image
from multiprocessing import Process, Queue
from PIL import Image
import time

sys.path.append('/playpen-nas-ssd/awang/semantic_editing/mystyle/third_party/eg3d')
import dnnlib
import legacy
import torch
import numpy as np
from training.projector.camera_utils import LookAtPoseSampler


device = torch.device('cuda')
id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def get_lookat_pose_sampler(camera_lookat_point, device):
    w_frames = 120
    frame_idx_1 = np.random.randint(0, w_frames)
    frame_idx_2 = w_frames - frame_idx_1 - 1
    pitch_range = 0.25
    yaw_range = 0.35
    num_keyframes = 1
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx_1 / (num_keyframes * w_frames)),
                                            camera_lookat_point, radius=2.7, device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)
    return c

def interpolate_subprocess(queue, pairs, celeb, year):
    # get features from reference set and memoize in a dictionary
    reference_set_loc = os.path.join('/playpen-nas-ssd/awang/data', celeb, 'all', 'all')
    reference_set_imgs = [os.path.join(reference_set_loc, x) for x in os.listdir(reference_set_loc) if x.endswith('.png')]
    reference_set_features = {}
    for img_path in reference_set_imgs:
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        reference_set_features[img_path] = person_identifier.get_feature(img)\

    for model_name in ['upper_bound', 'lower_bound']:
    #for model_name in ['lora_expansion', 'lora_expansion_baseline']:
        network_pkl = os.path.join('networks', celeb, model_name + '.pkl')
        with dnnlib.util.open_url(network_pkl) as f:
            model = legacy.load_network_pkl(f)['G_ema'].to(device)
        sum_of_means = 0
        count = 0
        for pair in pairs:
            latent_code_1 = np.load(pair[0])
            camera_param_1 = np.load(pair[0].replace('_latent', '')).reshape(1,25)
            latent_code_2 = np.load(pair[1])
            camera_param_2 = np.load(pair[1].replace('_latent', '')).reshape(1,25)

            pair_dir = os.path.join('interpolated', celeb, year, model_name, os.path.basename(pair[0])[:-11] + '_' + os.path.basename(pair[1])[:-11])
            interpolated_imgs = []
            for i in range(11):  # Interpolate at 10 steps
                t = i / 10.0
                interpolated_w = latent_code_1 * (1 - t) + latent_code_2 * t
                interpolated_c = camera_param_1 * (1 - t) + camera_param_2 * t
                ws = torch.from_numpy(interpolated_w).to(device)
                c = torch.from_numpy(interpolated_c).to(device)
                interp_img = model.synthesis(ws, c, noise_mode='const')['image'][0].unsqueeze(0)
                interpolated_imgs.append(interp_img)

                save_dir = os.path.join(pair_dir, f'{t}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                sum_errors_for_step = 0
                errors_dict = {}
                num_poses = 10
                for j in range(num_poses): # get 10 different novel views
                    random_c = get_lookat_pose_sampler(camera_lookat_point=torch.tensor([0, 0, 0], device='cuda'), device='cuda')
                    imgs = model.synthesis(ws, random_c, noise_mode='const')['image'][0].unsqueeze(0)
                    save_image(imgs, os.path.join(save_dir, f'{j}.png'), nrow=1, normalize=True, range=(-1, 1))
                    imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0]
                    feature = person_identifier.get_feature(imgs)
                    sims = {k: person_identifier.compute_similarity(feature, v) for k, v in reference_set_features.items()}
                    sim = max(sims.values()).item() # similarity to closest neighbor
                    id_error = 1 - sim
                    errors_dict[j] = id_error
                    sum_errors_for_step += id_error
                
                mean_error_for_step = sum_errors_for_step / num_poses
                sum_of_means += mean_error_for_step
                count += 1

                # Save errors_dict in the directory for i
                errors_dict_path = os.path.join(save_dir, f'errors_dict_{i}.json')
                with open(errors_dict_path, 'w') as f:
                    json.dump(errors_dict, f)

            # Save interpolated images
            interpolated_imgs = torch.cat(interpolated_imgs, dim=0)
            save_image(interpolated_imgs, os.path.join(pair_dir, 'interpolation.png'), nrow=11, normalize=True, range=(-1, 1))
        queue.put((model_name, sum_of_means, count))
        
        
def interpolate_each_time(celeb, year):
    # check to see if interpolation is already done
    for model_name in ['lower_bound', 'upper_bound']:
    #for model_name in ['lora_expansion', 'lora_expansion_baseline']:
        result_save_path = os.path.join('interpolated', celeb, year, model_name, 'mean_id_error.json')
        if os.path.exists(result_save_path):
            print(f'Interpolation for {celeb} {year} {model_name} already done. Skipping.')
            return

    latent_codes_dir = os.path.join('/playpen-nas-ssd/awang/data', celeb, year, 'train', 'preprocessed')
    latent_codes_locs = [os.path.join(latent_codes_dir, x) for x in os.listdir(latent_codes_dir) if 'latent' in x and 'mirror' not in x]
    combinations = list(itertools.combinations(latent_codes_locs, 2))
    shuffle(combinations)
    pairs = combinations[:10]

    num_processes = 1
    chunk_length = len(pairs) // num_processes
    # split pairs into chunks of length chunk_length
    chunks = [pairs[i:i+chunk_length] for i in range(0, len(pairs), chunk_length)]
    print('chunks:', chunks)

    q = Queue()
    processes = []
    rets = []
    
    try:
        for chunk in chunks:
            p = Process(target=interpolate_subprocess, args=(q, chunk, celeb, year, ))
            processes.append(p)
            p.start()
        for p in processes:
            ret = q.get() # will block
            rets.append(ret)
        for p in processes:
            p.join()
        for model_name in ['lower_bound', 'upper_bound']:
            overall_sum = 0
            overall_count = 0
            for ret in rets:
                if ret[0] == model_name:
                    overall_sum += ret[1]
                    overall_count += ret[2]
            avg_mean_error = overall_sum / overall_count
            result_save_path = os.path.join('interpolated', celeb, year, model_name, 'mean_id_error.json')
            with open(result_save_path, 'w') as f:
                json.dump({'mean_id_error': avg_mean_error}, f)
        
    except Exception as e:
        notify(f'Error in interpolate_each_time: {e}')
    
def interpolate_using_all_anchors(celeb):
    latent_codes_dir = os.path.join('/playpen-nas-ssd/awang/data', celeb, 'all', 'train', 'preprocessed')
    latent_codes_locs = [os.path.join(latent_codes_dir, x) for x in os.listdir(latent_codes_dir) if 'latent' in x and 'mirror' not in x]
    combinations = list(itertools.combinations(latent_codes_locs, 2))
    shuffle(combinations)
    pairs = combinations[:10]

    num_processes = 3
    chunk_length = len(pairs) // num_processes
    # split pairs into chunks of length chunk_length
    chunks = [pairs[i:i+chunk_length] for i in range(0, len(pairs), chunk_length)]

    q = Queue()
    processes = []
    rets = []
    
    try:
        for chunk in chunks:
            p = Process(target=interpolate_subprocess, args=(q, chunk, celeb, 'all', ))
            processes.append(p)
            p.start()
        for p in processes:
            ret = q.get() # will block
            rets.append(ret)
        for p in processes:
            p.join()
        for model_name in ['lower_bound', 'upper_bound']:
            overall_sum = 0
            overall_count = 0
            for ret in rets:
                if ret[0] == model_name:
                    overall_sum += ret[1]
                    overall_count += ret[2]
            avg_mean_error = overall_sum / overall_count
            result_save_path = os.path.join('interpolated', celeb, 'all', model_name, 'mean_id_error.json')
            with open(result_save_path, 'w') as f:
                json.dump({'mean_id_error': avg_mean_error}, f)
        
    except Exception as e:
        notify(f'Error in interpolate_using_all_anchors: {e}')   
        


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser() 
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--celeb', type=str, required=True)

    args = parser.parse_args()
    celeb = args.celeb
    t = args.year

    start_time = time.time()  # Record the start time
    interpolate_each_time(celeb, t)

    end_time = time.time()  # Record the end time
    execution_time = (end_time - start_time) / 60.0  # Calculate the execution time
    print(f"Execution time: {execution_time} minutes")
    notify(f'Finished interpolating {celeb} {t}. Took {execution_time} minutes.')
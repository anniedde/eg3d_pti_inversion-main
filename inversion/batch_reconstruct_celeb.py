import os
import multiprocessing
import argparse
import requests
import time
import torch
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def run_subprocess(celeb, year, gpu, experiment, model):
    subprocess_cmd = f"python reconstruct_celeb.py --year={year} --gpu={gpu} --celeb={celeb} --experiment={experiment} --model={model}"
    os.system(subprocess_cmd)

def compile_results(celeb):
    save_img_name_map = {
        'final_rgb_proj.png': 'pti',
        'before_pti_rgb_proj.png': 'before_pti',
        'input.png': 'input'
    }
    models = os.listdir(os.path.join('embeddings', celeb))
    for model in models:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--which_gpus', type=str, required=False)
    parser.add_argument('--experiments', type=str, required=False, default=None)
    parser.add_argument('--models', type=str, required=False, default='0,1,2,3,4,5,6,7,8,9')

    args = parser.parse_args()
    celeb = args.celeb
    models = args.models.split(',')
    num_gpus = torch.cuda.device_count()
    if args.which_gpus:
        gpus = args.which_gpus.split(',')
        num_gpus = len(gpus)
    else:
        gpus = range(num_gpus)

    networks_folder = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}'
    if args.experiments:
        experiments = args.experiments.split(',')
    else:
        experiments = [e for e in os.listdir(networks_folder) if os.path.isdir(os.path.join(networks_folder, e))]
    
    data_path = f'/playpen-nas-ssd/awang/data/{celeb}'

    start = time.time()
    try:
        for experiment in experiments:
            models.sort(reverse=True)
            if experiment == 'upper_bound':
                model = 'all'
                years = range(10)

                year_chunks = [years[i:i + num_gpus] for i in range(0, len(years), num_gpus)]
                processes = []
                for chunk in year_chunks:
                    for i, year in enumerate(chunk):
                        p = multiprocessing.Process(target=run_subprocess, args=(celeb, year, gpus[i], experiment, model, ))
                        p.start()
                        processes.append(p)

                    for p in processes:
                        p.join()
            else:
                for model in models:
                    years = range(0, int(model) + 1)
                    year_chunks = [years[i:i + num_gpus] for i in range(0, len(years), num_gpus)]

                    processes = []
                    for chunk in year_chunks:
                        for i, year in enumerate(chunk):
                            p = multiprocessing.Process(target=run_subprocess, args=(celeb, year, gpus[i], experiment, model, ))
                            p.start()
                            processes.append(p)

                        for p in processes:
                            p.join()

        #compile_results(celeb)
                    
    except Exception as e:
        notify(f"Error in reconstruction for {celeb}: {e}")

    end = time.time()
    execution_time = (end - start) / 60
    notify(f"Batch reconstruction for {celeb} complete. Took {execution_time} minutes.")
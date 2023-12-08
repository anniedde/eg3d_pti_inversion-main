import os
import multiprocessing
import argparse
import requests
import time
from interpolate_celeb import interpolate_using_all_anchors
import torch

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def run_subprocess(celeb, year, gpu):
    subprocess_cmd = f'CUDA_VISIBLE_DEVICES={gpu} python interpolate_celeb.py --celeb={celeb} --year={year}' 
    os.system(subprocess_cmd)



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--which_gpus', type=str, required=False)

    args = parser.parse_args()
    celeb = args.celeb

    if args.which_gpus:
        gpus = args.which_gpus.split(',') 
        num_gpus = len(gpus)
    else:
        num_gpus = torch.cuda.device_count()
        gpus = range(num_gpus)
    
    years = list(range(10))
    year_chunks = [years[i:i + num_gpus] for i in range(0, len(years), num_gpus)]

    start = time.time()
    try:
        
        processes = []
        for chunk in year_chunks:
            for i, year in enumerate(chunk):
                p = multiprocessing.Process(target=run_subprocess, args=(celeb, year, gpus[i],))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        interpolate_using_all_anchors(celeb)

    except Exception as e:
        notify(f"Error in interpolation for {celeb}: {e}")

    end = time.time()
    execution_time = (end - start) / 60
    notify(f"Batch interpolation for {celeb} complete. Took {execution_time} minutes.")
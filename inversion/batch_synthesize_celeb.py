import os
import multiprocessing
import argparse
import requests
from synthesize_celeb import evaluate_across_years
import traceback
import time
import torch

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message

def run_generate_script_by_time(celeb, t, gpu='0'):
    for model_name in ['lower_bound', 'upper_bound']:
        command = 'conda run -n mystyle_new python generate.py ' \
                + f'--anchors_path=/playpen-nas-ssd/awang/data/{celeb}/{t}/train/preprocessed ' \
                + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{model_name}.pkl ' \
                + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/{t}/{model_name} ' \
                + f'--device={gpu}'
        os.system(command)

def run_generate_script_all(celeb, gpu='0'):
    for model_name in ['lower_bound', 'upper_bound']:
        command = 'conda run -n mystyle_new python generate.py ' \
                + f'--anchors_path=/playpen-nas-ssd/awang/data/{celeb}/all/train/ ' \
                + f'--generator_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb}/{model_name}.pkl ' \
                + f'--output_path=/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/{celeb}/all/{model_name} ' \
                + f'--device={gpu}'
        os.system(command)

def run_subprocess(celeb, year, gpu):
    subprocess_cmd = f'CUDA_VISIBLE_DEVICES={gpu} python synthesize_celeb.py --celeb={celeb} --year={year}'
    print(subprocess_cmd)
    os.system(subprocess_cmd)


if __name__ == "__main__":
    try:
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
        
        try: 
            # first generate all the synthetic images
            start = time.time()
            os.chdir("../../semantic_editing/mystyle")
            processes = []
            for chunk in year_chunks:
                for i, year in enumerate(chunk):
                    p = multiprocessing.Process(target=run_generate_script_by_time, args=(celeb, year, gpus[i],))
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()
            
            run_generate_script_all(celeb, gpu=gpus[0])
            os.chdir("../../eg3d_pti_inversion-main/inversion")
            
            # then get the id error for each synthesized image
            evaluate_across_years(celeb)
            processes = []
            for chunk in year_chunks:
                for i, year in enumerate(chunk):
                    p = multiprocessing.Process(target=run_subprocess, args=(celeb, year, gpus[i],))
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()
            
        except Exception as e:
            notify(f"Error in synthesis for {celeb}: {e}")

        end = time.time()
        execution_time = (end - start) / 60
        notify(f"Batch synthesis for {celeb} complete. Took {execution_time} minutes")
        
    except Exception as e:
        traceback.print_exc()

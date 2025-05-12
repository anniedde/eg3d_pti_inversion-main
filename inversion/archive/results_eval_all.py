import os
import sys
import subprocess
import shutil
import time

names = ['Barack','Michelle','Joe','Oprah', 'Taylor']
for name in names:
    configs_dir = '/playpen-nas-ssd/luchao/projects/eg3d/eg3d_pti_inversion/inversion/configs'
    for run_name in ['sym', 'baseline', 'pretrained', 'all', 'lora']:
        config_file_name = f'paths_config_{name.lower()}_{run_name}.py'
        config_file_path = os.path.join(configs_dir, config_file_name)
        # copy the config file to paths_config.py
        shutil.copy(config_file_path, os.path.join(configs_dir, 'paths_config.py'))
        # wait for 3 seconds
        time.sleep(3)
        os.makedirs('./out', exist_ok=True)
        os.makedirs('./latent_code', exist_ok=True)
        # run the script
        cmd1 = 'CUDA_VISIBLE_DEVICES=2 python gen_videos_all.py'
        process1 = subprocess.Popen(cmd1, shell=True)

        # cmd2 = 'CUDA_VISIBLE_DEVICES=3 python eval_interpolation.py'
        # process2 = subprocess.Popen(cmd2, shell=True)
        process1.wait()
        # process2.wait()

        # results are in out dir -> move out to new dest dir

        # for process1
        dest_dir = f'/playpen-nas-ssd/luchao/projects/eg3d/eg3d_pti_inversion/inversion/results/reconstruction_{name.lower()}_{run_name}'
        shutil.move('/playpen-nas-ssd/luchao/projects/eg3d/eg3d_pti_inversion/inversion/out', dest_dir)

        # for process2
        dest_dir = f'/playpen-nas-ssd/luchao/projects/eg3d/eg3d_pti_inversion/inversion/results/interpolation_{name.lower()}_{run_name}'
        shutil.move('/playpen-nas-ssd/luchao/projects/eg3d/eg3d_pti_inversion/inversion/latent_code', dest_dir)
        
        # done
        print('done with {} {}'.format(name, run_name))
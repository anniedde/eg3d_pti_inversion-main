import os
import time

try:
    while True:
        os.system('/playpen-nas-ssd/awang/my3dgen/eg3d/temp.sh')
        os.system('python batch_reconstruct_celeb_new.py --celeb Michael --experiments random --which_gpus 0,1,2,3')
        os.system('MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python batch_synthesize_celeb.py --celeb Michael --experiments random --which_gpus 0')
        os.system('python eval_reconstruction.py --celeb Michael --experiment ransac')
        os.system('python eval_reconstruction.py --celeb Michael --experiment random')

        time.sleep(10) #make function to sleep for 10 seconds
except KeyboardInterrupt:
    exit()
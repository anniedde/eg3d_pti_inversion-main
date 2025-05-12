import os
import time
os.environ['MKL_THREADING_LAYER'] = 'GNU'
try:
    while True:
        os.system('/playpen-nas-ssd/awang/my3dgen/eg3d/temp.sh')
        os.system('python batch_reconstruct_celeb_new.py --celeb Michael --experiments ransac --which_gpus 0,1,2,3,4,5,6,7')
        os.system('MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python batch_synthesize_celeb.py --celeb Michael --experiments ransac --which_gpus 0')

        time.sleep(10) #make function to sleep for 10 seconds
except KeyboardInterrupt:
    exit()
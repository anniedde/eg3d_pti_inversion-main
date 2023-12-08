import os
import sys
import subprocess
import argparse

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--which_gpus', type=str, required=False)
    args = parser.parse_args()

    if args.which_gpus:
        os.system('python batch_reconstruct_celeb.py --celeb={args.celeb} -u -l --which_gpus={args.which_gpus}')
        os.system('python batch_interpolate_celeb.py --celeb={args.celeb} --which_gpus={args.which_gpus}')
        os.system('python batch_synthesize_celeb.py --celeb={args.celeb} --which_gpus={args.which_gpus}')
    else:
        os.system('python batch_reconstruct_celeb.py --celeb={args.celeb} -u -l')
        os.system('python batch_interpolate_celeb.py --celeb={args.celeb}')
        os.system('python batch_synthesize_celeb.py --celeb={args.celeb}')

    os.system('python graph_results.py --embedding_folder={args.celeb}')
    
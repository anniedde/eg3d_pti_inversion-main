import os
import argparse
import shutil

def move_networks(celeb_name, replay_method):
    dest_dir = f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/{celeb_name}/{replay_method}'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    training_runs_dir = f'/playpen-nas-ssd/awang/my3dgen/eg3d/training-runs/replay/{celeb_name}_{replay_method}'
    assert os.path.exists(training_runs_dir), f'{training_runs_dir} does not exist'

    for vid_folder in os.listdir(training_runs_dir):
        vid_folder_path = os.path.join(training_runs_dir, vid_folder)
        
        vid_num = vid_folder.split('-')[-1][3:]

        if os.path.exists(os.path.join(vid_folder_path, 'network-snapshot-000500.pkl')):
            shutil.copy(os.path.join(vid_folder_path, 'network-snapshot-000500.pkl'), os.path.join(dest_dir, f'{vid_num}.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--celeb", type=str, required=True)
    parser.add_argument('--num_gpus', type=int, default=4)
    parser.add_argument('--which_gpus', type=str, required=True)
    parser.add_argument('--experiments', type=str, required=False, default=None)
    parser.add_argument('--models', type=str, required=False, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--years', type=str, required=False, default=None)
    args = parser.parse_args()

    try:
        for experiment in args.experiments.split(','):
            move_networks(args.celeb, experiment)
        if args.which_gpus:
            os.system(f'python batch_reconstruct_celeb.py --celeb={args.celeb} --which_gpus={args.which_gpus} --models={args.models} --experiments={args.experiments}')
            #os.system(f'python batch_interpolate_celeb.py --celeb={args.celeb} --which_gpus={args.which_gpus}')
            os.system(f'MKL_NUM_THREADS=1 python batch_synthesize_celeb.py --celeb={args.celeb} --which_gpus={args.which_gpus} --models={args.models} --experiments={args.experiments} --years={args.years}')
        else:
            os.system(f'python batch_reconstruct_celeb.py --celeb={args.celeb}  --models={args.models} --experiments={args.experiments}')
            #os.system(f'python batch_interpolate_celeb.py --celeb={args.celeb}')
            os.system(f'MKL_NUM_THREADS=1 python batch_synthesize_celeb.py --celeb={args.celeb} --models={args.models} --experiments={args.experiments} --years={args.years}')

        #os.system(f'python graph_results.py --celeb={args.celeb} --models={args.models}')
    except KeyboardInterrupt as e:
        print(e)
        print('KeyboardInterrupt')
        exit()
import os, shutil, json
import numpy as np
import argparse

def load_recon_metrics(metrics_loc):
    assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"
    with open(metrics_loc, 'r') as f:
        metrics = json.load(f)
        lpips = metrics['lpips']
        psnr = metrics['psnr']
        dists = metrics['dists']
        if 'id_sim' not in metrics:
            id_sim = 1 - metrics['id_error']
        else:
            id_sim = metrics['id_sim']
    return lpips, psnr, dists, id_sim

def load_synthesis_metrics(metrics_loc):
    assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"
    with open(metrics_loc, 'r') as f:
        metrics = json.load(f)
        fid = metrics['fid']
        if 'id_sim' not in metrics:
            id_sim = 1 - metrics['mean_id_error']
        else:
            id_sim = metrics['id_sim']
    return fid, id_sim

def get_forgetting(celeb, experiment, j):
    # model trained until time 9
    # get forgetting on data cluster j
    lpips_list, psnr_list, dists_list, id_sim_list = [], [], [], []

    if experiment == 'upper_bound':
        raise Exception('Forgetting does not apply to upper bound')
    else:
        max_diff_lpips, max_diff_psnr, max_diff_dists, max_diff_id_sim = 0, 0, 0, 0
        
        # get average performance of model 9 on time j
        lpips_T, psnr_T, dists_T, id_sim_T = 0, 0, 0, 0
        for image_id in range(10):
            metrics_loc_T = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment, '19', str(j), str(image_id), 'metrics.json')
            lpips, psnr, dists, id_sim = load_recon_metrics(metrics_loc_T)
            lpips_T += lpips
            psnr_T += psnr
            dists_T += dists
            id_sim_T += id_sim
        lpips_T /= 10
        psnr_T /= 10
        dists_T /= 10
        id_sim_T /= 10
        
        for l in range(j, 19):
            # get difference between model trained until time l and model trained until time 9
            metrics_loc_l = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment, str(l), str(j), 'metrics.json')
            # get average performance of model 9 on time j
            lpips_l, psnr_l, dists_l, id_sim_l = 0, 0, 0, 0
            for image_id in range(10):
                metrics_loc_l = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment, str(l), str(j), str(image_id), 'metrics.json')
                lpips, psnr, dists, id_sim = load_recon_metrics(metrics_loc_l)
                lpips_l += lpips
                psnr_l += psnr
                dists_l += dists
                id_sim_l += id_sim
            lpips_l /= 10
            psnr_l /= 10
            dists_l /= 10
            id_sim_l /= 10
            
            diff_lpips = lpips_T - lpips_l
            diff_psnr = psnr_l - psnr_T
            diff_dists = dists_T - dists_l
            diff_id_sim = id_sim_l - id_sim_T

            lpips_list.append(diff_lpips)
            psnr_list.append(diff_psnr)
            dists_list.append(diff_dists)
            id_sim_list.append(diff_id_sim)
        
        max_diff_lpips = max(lpips_list)
        max_diff_psnr = max(psnr_list)
        max_diff_dists = max(dists_list)
        max_diff_id_sim = max(id_sim_list)

        return max_diff_lpips, max_diff_psnr, max_diff_dists, max_diff_id_sim

def get_average_forgetting(celeb, experiment):
    lpips_list, psnr_list, dists_list, id_sim_list = [], [], [], []
    if experiment == 'upper_bound':
        raise Exception('Forgetting does not apply to upper bound')

    for t in range(19):
        lpips, psnr, dists, id_sim = get_forgetting(celeb, experiment, t)
        lpips_list.append(lpips)
        psnr_list.append(psnr)
        dists_list.append(dists)
        id_sim_list.append(id_sim)

    average_forgetting = {
        'lpips': np.mean(lpips_list),
        'psnr': np.mean(psnr_list),
        'dists': np.mean(dists_list),
        'id_sim': np.mean(id_sim_list)
    }
    print('Average reconstruction forgetting for celeb: {}, experiment: {}'.format(celeb, experiment))
    print(average_forgetting)

    # save average performance to json
    with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment, 'average_forgetting.json'), 'w') as f:
        json.dump(average_forgetting, f)

def get_forgetting_synthesis(celeb, experiment, j):
    # model trained until time t
    fid_list, id_sim_list = [], []
    metrics_loc_T = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/', celeb, experiment, '19', str(j), 'metrics.json')
    fid_T, id_sim_T = load_synthesis_metrics(metrics_loc_T)
    for l in range(j, 19):
        # get difference between model trained until time l and model trained until time 9
        metrics_loc_l = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/', celeb, experiment, str(l), str(j), 'metrics.json')
        fid_l, id_sim_l = load_synthesis_metrics(metrics_loc_l)
        diff_fid = fid_T - fid_l
        diff_id_sim = id_sim_l - id_sim_T
        
        fid_list.append(diff_fid)
        id_sim_list.append(diff_id_sim)

    max_diff_fid = max(fid_list)
    max_diff_id_sim = max(id_sim_list)
    
    return max_diff_fid, max_diff_id_sim

def get_average_forgetting_synthesis(celeb, experiment):
    fid_list, id_sim_list = [], []
    if experiment == 'upper_bound':
        raise Exception('Forgetting does not apply to upper bound')

    for t in range(19):
        fid, id_sim = get_forgetting_synthesis(celeb, experiment, t)
        fid_list.append(fid)
        id_sim_list.append(id_sim)

    average_forgetting = {
        'fid': np.mean(fid_list),
        'id_sim': np.mean(id_sim_list)
    }
    print('Average synthesis forgetting for celeb: {}, experiment: {}'.format(celeb, experiment))
    print(average_forgetting)

    # save average performance to json
    with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/', celeb, experiment, 'average_forgetting.json'), 'w') as f:
        json.dump(average_forgetting, f)

def process_args():
    parser = argparse.ArgumentParser(description='Get average performance of model t on time k')
    parser.add_argument('--celeb', type=str, help='celebrity name')
    parser.add_argument('--experiment', type=str, help='experiment name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = process_args()


    get_average_forgetting(args.celeb, args.experiment)
    get_average_forgetting_synthesis(args.celeb, args.experiment)

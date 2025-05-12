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
            metrics_loc_T = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/', celeb, experiment, '9', str(j), str(image_id), 'metrics.json')
            lpips, psnr, dists, id_sim = load_recon_metrics(metrics_loc_T)
            lpips_T += lpips
            psnr_T += psnr
            dists_T += dists
            id_sim_T += id_sim
        lpips_T /= 10
        psnr_T /= 10
        dists_T /= 10
        id_sim_T /= 10
        
        for l in range(j, 9):
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

    return average_forgetting

def get_forgetting_synthesis(celeb, experiment, j):
    # model trained until time t
    fid_list, id_sim_list = [], []
    metrics_loc_T = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized/', celeb, experiment, '9', str(j), 'metrics.json')
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

    return average_forgetting

def get_average_performance(celeb, experiment, t):
    # model trained until time t
    lpips_list, psnr_list, dists_list, id_sim_list = [], [], [], []

    if experiment == 'upper_bound':
        for k in range(10):
            # get average performance of model t on time k
            for image_id in range(10):
                metrics_loc = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', celeb, experiment, 'all', str(k), str(image_id), 'metrics.json')
                assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"

                with open(metrics_loc, 'r') as f:
                    metrics = json.load(f)
                    lpips = metrics['lpips']
                    psnr = metrics['psnr']
                    dists = metrics['dists']
                    id_sim = metrics['id_sim']
                    lpips_list.append(lpips)
                    psnr_list.append(psnr)
                    dists_list.append(dists)
                    id_sim_list.append(id_sim)
    else:
        for k in range(t + 1):
            # get average performance of model t on time k
            lpips, psnr, dists, id_sim = 0, 0, 0, 0
            for image_id in range(10):
                metrics_loc = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', celeb, experiment, str(t), str(k), str(image_id), 'metrics.json')
                assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"

                with open(metrics_loc, 'r') as f:
                    metrics = json.load(f)
                    lpips += metrics['lpips']
                    psnr += metrics['psnr']
                    dists += metrics['dists']
                    id_sim += metrics['id_sim']
            lpips_list.append(lpips / 10)
            psnr_list.append(psnr / 10)
            dists_list.append(dists / 10)
            id_sim_list.append(id_sim / 10)

    average_performance = {
        'lpips': np.mean(lpips_list),
        'psnr': np.mean(psnr_list),
        'dists': np.mean(dists_list),
        'id_sim': np.mean(id_sim_list)
    }

    # save average performance to json
    #with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', celeb, experiment, str(t), 'average_performance.json'), 'w') as f:
    #    json.dump(average_performance, f)

    return np.mean(lpips_list), np.mean(psnr_list), np.mean(dists_list), np.mean(id_sim_list)

def get_average_incremental_performance(celeb, experiment):
    lpips_list, psnr_list, dists_list, id_sim_list = [], [], [], []
    if experiment == 'upper_bound':
        for t in ['all']:
            lpips, psnr, dists, id_sim = get_average_performance(celeb, experiment, t)
            lpips_list.append(lpips)
            psnr_list.append(psnr)
            dists_list.append(dists)
            id_sim_list.append(id_sim)
    else:
        for t in range(20):
            lpips, psnr, dists, id_sim = get_average_performance(celeb, experiment, t)
            lpips_list.append(lpips)
            psnr_list.append(psnr)
            dists_list.append(dists)
            id_sim_list.append(id_sim)

    average_incremental_performance = {
        'lpips': np.mean(lpips_list),
        'psnr': np.mean(psnr_list),
        'dists': np.mean(dists_list),
        'id_sim': np.mean(id_sim_list)
    }
    print('Average incremental reconstruction performance for celeb: {}, experiment: {}'.format(celeb, experiment))
    print(average_incremental_performance)

    # save average performance to json
    with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', celeb, experiment, 'average_incremental_performance.json'), 'w') as f:
        json.dump(average_incremental_performance, f)

    return average_incremental_performance

def get_average_performance_synthesis(celeb, experiment, t):
    # model trained until time t
    fid_list, id_sim_list = [], []

    if experiment == 'upper_bound':
        for k in range(10):
            # get average performance of model t on time k
            metrics_loc = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized', celeb, experiment, 'all', str(k), 'metrics.json')
            assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"

            with open(metrics_loc, 'r') as f:
                metrics = json.load(f)
            fid = metrics['fid']
            if 'id_sim' not in metrics:
                id_sim = 1 - metrics['mean_id_error']
            else:
                id_sim = metrics['id_sim']
            id_sim_list.append(id_sim)
            fid_list.append(fid)
    else:
        for k in range(t + 1):
            # get average performance of model t on time k
            metrics_loc = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized', celeb, experiment, str(t), str(k), 'metrics.json')
            assert os.path.exists(metrics_loc), f"metrics.json not found at {metrics_loc}"

            with open(metrics_loc, 'r') as f:
                metrics = json.load(f)
            fid = metrics['fid']
            if 'id_sim' not in metrics:
                id_sim = 1 - metrics['mean_id_error']
            else:
                id_sim = metrics['id_sim']
            id_sim_list.append(id_sim)
            fid_list.append(fid)

    average_performance = {
        'fid': np.mean(fid_list),
        'id_sim': np.mean(id_sim_list)
    }

    # save average performance to json
    #with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', celeb, experiment, str(t), 'average_performance.json'), 'w') as f:
    #    json.dump(average_performance, f)

    return np.mean(fid_list), np.mean(id_sim_list)

def get_average_incremental_performance_synthesis(celeb, experiment):
    fid_list, id_sim_list = [], []
    if experiment == 'upper_bound':
        for t in ['all']:
            fid, id_sim = get_average_performance_synthesis(celeb, experiment, t)
            fid_list.append(fid)
            id_sim_list.append(id_sim)
    else:
        for t in range(20):
            fid, id_sim = get_average_performance_synthesis(celeb, experiment, t)
            fid_list.append(fid)
            id_sim_list.append(id_sim)

    average_incremental_performance = {
        'fid': np.mean(fid_list),
        'id_sim': np.mean(id_sim_list)
    }
    print('Average incremental synthesis performance for celeb: {}, experiment: {}'.format(celeb, experiment))
    print(average_incremental_performance)

    # save average performance to json
    with open(os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized', celeb, experiment, 'average_incremental_performance.json'), 'w') as f:
        json.dump(average_incremental_performance, f)

    return average_incremental_performance

def process_args():
    parser = argparse.ArgumentParser(description='Get average performance of model t on time k')
    parser.add_argument('--celeb', type=str, help='celebrity name')
    parser.add_argument('--experiment', type=str, help='experiment name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = process_args()

    if args.celeb:
        get_average_incremental_performance(args.celeb, args.experiment)
        get_average_incremental_performance_synthesis(args.celeb, args.experiment)
    else:
        celebs = ['Margot', 'Harry', 'IU', 'Michael']

        vals = []
        aip_fids, aip_id_sims, forgetting_fids, forgetting_id_sims = [], [], [], []
        for celeb in celebs:
            try:
                aip_synthesis = get_average_incremental_performance_synthesis(celeb, args.experiment)
                forgetting_synthesis = get_average_forgetting_synthesis(celeb, args.experiment)
            except:
                aip_synthesis = {'fid': -1000000, 'id_sim': -1000000}
                forgetting_synthesis = {'fid': -1000000, 'id_sim': -1000000}

            
            # round fid to first decimal place
            aip_fid = round(aip_synthesis['fid'], 1)
            forgetting_fid = round(forgetting_synthesis['fid'], 1)
            aip_id_sim = round(aip_synthesis['id_sim'] * 10, 2)
            forgetting_id_sim = round(forgetting_synthesis['id_sim'] * 10, 2)
            
            vals.append(aip_fid)
            vals.append(aip_id_sim)
            vals.append(forgetting_fid)
            vals.append(forgetting_id_sim)

            aip_fids.append(aip_fid)
            aip_id_sims.append(aip_id_sim)
            forgetting_fids.append(forgetting_fid)
            forgetting_id_sims.append(forgetting_id_sim)

        row_string_synthesis = ' & '.join([str(val) for val in vals])
        row_string_synthesis = row_string_synthesis.replace('-1000000', 'x.xx')
        mean_string_synthesis = ' & '.join([str(round(np.mean(aip_fids), 1)), \
                                            str(round(np.mean(aip_id_sims), 2)), \
                                            str(round(np.mean(forgetting_fids), 1)), \
                                            str(round(np.mean(forgetting_id_sims), 2))])
        row_string_synthesis = row_string_synthesis + ' & ' + mean_string_synthesis

        vals = []
        aip_lpips_list, aip_id_sims, forgetting_lpips_list, forgetting_id_sims = [], [], [], []
        for celeb in celebs:
            try:
                aip_recon = get_average_incremental_performance(celeb, args.experiment)
                forgetting_recon = get_average_forgetting(celeb, args.experiment)
            except:
                aip_recon = {'lpips': -1000000, 'psnr': -1000000, 'dists': -1000000, 'id_sim': -1000000}
                forgetting_recon = {'lpips': -1000000, 'psnr': -1000000, 'dists': -1000000, 'id_sim': -1000000}

            aip_lpips = round(aip_recon['lpips'] * 10, 2)
            aip_id_sim = round(aip_recon['id_sim'] * 10, 2)
            forgetting_lpips = round(forgetting_recon['lpips'] * 10, 2)
            forgetting_id_sim = round(forgetting_recon['id_sim'] * 10, 2)

            vals.append(aip_lpips)
            vals.append(aip_id_sim)
            vals.append(forgetting_lpips)
            vals.append(forgetting_id_sim)

            aip_lpips_list.append(aip_lpips)
            aip_id_sims.append(aip_id_sim)
            forgetting_lpips_list.append(forgetting_lpips)
            forgetting_id_sims.append(forgetting_id_sim)

        row_string = ' & '.join([str(val) for val in vals])
        row_string = row_string.replace('-1000000', 'x.xx')

        mean_string = ' & '.join([str(round(np.mean(aip_lpips_list), 2)), \
                                    str(round(np.mean(aip_id_sims), 2)), \
                                    str(round(np.mean(forgetting_lpips_list), 2)), \
                                    str(round(np.mean(forgetting_id_sims), 2))])
        row_string = row_string + ' & ' + mean_string
        print(f'3D recon row for {args.experiment}: {row_string}')

        print(f'3D synthesis row for {args.experiment}: {row_string_synthesis}')


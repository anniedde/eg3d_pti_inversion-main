import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path, id_model, \
                                        networks_dir, embeddings_folder, evaluate_metrics
import json

device=torch.device('cuda')

def subplot(ax, map, metric, times):
    X = list(map.keys())  # Convert map keys to a list
    #X = ['lower_bound', 't0_upper_bound', 't0_t1_upper_bound', 'rank_expansion', 'replay_regular_lora_expansion', 'lora_svd_rank1_1step', 'replay_lora_svd_rank1']
    
    t0_values = [map[model][0] for model in X]
    t1_values = [map[model][1] for model in X]
    
    X_axis = np.arange(len(X)) 
    
    bars = ax.bar(X_axis, t0_values, 0.4, label='t = 0')
    # print the number values right below the top of each bar 
    for i, v in enumerate(t0_values):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')


    colors = ['b','g','r','c','m','y','k']
    #for i, (x, y) in enumerate(zip(X_axis, t1_values)):
    #    print(f'writing line for model {X[i]} in color {colors[i]}')
    #    ax.axhline(y, xmin=x-0.2, xmax=x+0.2, color=colors[i], linestyle='--', linewidth=1)
    #    ax.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    x_start = np.array([plt.getp(item, 'x') for item in bars])
    x_end   = x_start+[plt.getp(item, 'width') for item in bars]

    ax.hlines(t1_values, x_start, x_end, color='r')
    
    ax.set_xticks(X_axis)  # Set the x-axis tick positions
    ax.set_xticklabels(X, rotation=-45, ha="left")  # Set the x-axis tick labels
    ax.set(xlabel='Models', ylabel=metric) 
    ax.set_title(metric)
    ax.legend()

def graph_reconstruction_results(celeb, models_list=None, title='reconstruction_eval'):
    root_folder = os.path.join('embeddings', celeb)
    if not models_list:
        models_list = os.listdir(root_folder)
    lpips = LPIPS(net='alex').to(device).eval()
    dists = DISTS().to(device)
    person_identifier = id_utils.PersonIdentifier(id_model, None, None)

    lpips_map, psnr_map, dists_map, id_error_map = {}, {}, {}, {}
    for model_name in models_list:
        lpips_list, psnr_list, dists_list, id_error_list = [], [], [], []
        times = sorted(os.listdir(os.path.join(root_folder, model_name)))
        times = sorted(times, key=lambda x: (x[0].isdigit(), x))
        times = ['0','1']
        for t in times:
            embedding_folder = os.path.join(root_folder, model_name, t)
            print('celeb: {}, model: {}, time: {}'.format(celeb, model_name, t))
            lpips_loss, psnr, dists_loss, id_error = evaluate_metrics(embedding_folder, False, lpips, dists, person_identifier)
            lpips_list.append(lpips_loss)
            psnr_list.append(psnr)
            dists_list.append(dists_loss)
            id_error_list.append(id_error)
        
        lpips_map[model_name] = lpips_list
        psnr_map[model_name] = psnr_list
        dists_map[model_name] = dists_list
        id_error_map[model_name] = id_error_list

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.set_figwidth(13)
    fig.set_figheight(10)
    subplot(axs[0, 0], lpips_map, 'LPIPS', times)
    subplot(axs[0, 1], psnr_map, 'PSNR', times)
    subplot(axs[1, 0], dists_map, 'DISTS', times)
    subplot(axs[1, 1], id_error_map, 'ID Error', times)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.20)

    outdir = os.path.join('out', celeb)
    if not os.path.exists(outdir):
            os.makedirs(outdir)

    fig.suptitle('Reconstruction Evaluation without PTI for {}'.format(celeb))
    plt.savefig(os.path.join(outdir, f'{title}_reconstruction_plot_no_pti.png'))

    plt.clf()

def graph_synthesis_results(celeb, models=None, title='synthesis_eval'):
    # first do the graph for all times
    root_folder = os.path.join('synthesized', celeb)
    times = sorted([x for x in os.listdir(root_folder) if x != 'all'])
    #models = ['lower_bound', 't0_upper_bound', 't0_t1_upper_bound', 'rank_expansion', 'replay_regular_lora_expansion', 'lora_svd_rank1_1step', 'replay_lora_svd_rank1']
    
    errors = {}
    for t in times:
        for model in models:
            # read the json file
            with open(os.path.join(root_folder, t, model, 'fid_score.json'), 'r') as f:
                error = json.load(f)['fid']
                if model not in errors:
                    errors[model] = []
                errors[model].append(error)

    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    subplot(axs, errors, 'FID score', times)
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.20)

    outdir = os.path.join('out', celeb)
    if not os.path.exists(outdir):
            os.makedirs(outdir)

    fig.suptitle('Synthesis Results (each time separately) for {}'.format(celeb))
    plt.savefig(os.path.join(outdir, f'{title}_synthesis_separate.png'))

    plt.clf()

    # then make a bar graph comparing the models' performance overall 
    errors = []
    for model in models:
        # open json file
        with open(os.path.join(root_folder, 'all', model, 'fid_score.json'), 'r') as f:
            error = json.load(f)['fid']
            errors.append(error)
    plt.bar(models, errors)
    # rotate x axis labels
    plt.xticks(rotation=-45, ha="left")
    plt.xlabel('model')
    plt.ylabel('FID score')
    plt.title('Synthesis Evaluation Results for {} (using all anchors)'.format(celeb))
    plt.savefig(os.path.join('out', celeb, f'{title}_synthesis_plot_all_anchors.png'))
    plt.clf()

    """

    
    for model in models:
        plt.plot(range(len(times)), errors[model], '-o', label=model)

    t0_values = [errors[model][0] for model in models]
    t1_values = [errors[model][1] for model in models]
    
    X_axis = np.arange(len(X)) 
    
    bars = ax.bar(X_axis, t0_values, 0.4, label='t = 0')

    colors = ['b','g','r','c','m','y','k']
    #for i, (x, y) in enumerate(zip(X_axis, t1_values)):
    #    print(f'writing line for model {X[i]} in color {colors[i]}')
    #    ax.axhline(y, xmin=x-0.2, xmax=x+0.2, color=colors[i], linestyle='--', linewidth=1)
    #    ax.text(x, y, f'{y:.2f}', ha='center', va='bottom')

    x_start = np.array([plt.getp(item, 'x') for item in bars])
    x_end   = x_start+[plt.getp(item, 'width') for item in bars]

    ax.hlines(t1_values, x_start, x_end, color='r')
    
    ax.set_xticks(X_axis)  # Set the x-axis tick positions
    ax.set_xticklabels(X, rotation=-45, ha="left")  # Set the x-axis tick labels
    ax.set(xlabel='Models', ylabel=metric) 
    ax.set_title(metric)
    ax.legend()



    plt.legend()
    plt.xlabel('time')
    plt.ylabel('mean ID error (multiview)')
    plt.title('Synthesis Evaluation Results for {}'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'synthesis_plot_over_time.png'))
    plt.clf()

    # then make a bar graph comparing the models' performance overall 
    errors = []
    for model in models:
        # open json file
        with open(os.path.join(root_folder, 'all', model, 'mean_id_error.json'), 'r') as f:
            error = json.load(f)['mean_id_error']
            errors.append(error)
    plt.bar(models, errors)
    plt.xlabel('model')
    plt.ylabel('mean ID error (multiview)')
    plt.title('Synthesis Evaluation Results for {} (using all anchors)'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'synthesis_plot_all_anchors.png'))
    plt.clf()
    """

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, required=False, default='Margot_1step')
    parser.add_argument('--models', type=str, required=False)
    parser.add_argument('--title', type=str, required=False)
    args = parser.parse_args()
    celeb = args.celeb

    models_list = args.models.split(',') if args.models else None
    #models_list = ['lower_bound', 't0_upper_bound', 't0_t1_upper_bound', 'rank_expansion', 'replay_regular_lora_expansion', 'lora_svd_rank1_1step', 'replay_lora_svd_rank1']
    graph_reconstruction_results(celeb, models_list=models_list, title=args.title if args.title else 'synthesis_eval')
    graph_synthesis_results(celeb, models=models_list, title=args.title if args.title else 'synthesis_eval')

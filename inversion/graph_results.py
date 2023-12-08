import os
import argparse
import matplotlib.pyplot as plt
import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from utils_copy import id_utils
from utils.reconstruction_utils import paths_config_path, id_model, \
                                        networks_dir, embeddings_folder, evaluate_metrics
import json

device=torch.device('cuda')

def subplot(ax, map, metric):
    for model in map:
        length = len(map[model])
        ax.plot(range(length), map[model], '-o', label=model)
    ax.set(xlabel='time', ylabel=metric)
    ax.set_title(metric)
    ax.legend()

def graph_reconstruction_results(celeb):
    root_folder = os.path.join('embeddings', celeb)
    models_list = os.listdir(root_folder)
    lpips = LPIPS(net='alex').to(device).eval()
    dists = DISTS().to(device)
    person_identifier = id_utils.PersonIdentifier(id_model, None, None)

    for pti in [True, False]:
        lpips_map, psnr_map, dists_map, id_error_map = {}, {}, {}, {}
        for model_name in models_list:
            lpips_list, psnr_list, dists_list, id_error_list = [], [], [], []
            times = sorted(os.listdir(os.path.join(root_folder, model_name)))
            for t in times:
                embedding_folder = os.path.join(root_folder, model_name, t)
                lpips_loss, psnr, dists_loss, id_error = evaluate_metrics(embedding_folder, pti, lpips, dists, person_identifier)
                lpips_list.append(lpips_loss)
                psnr_list.append(psnr)
                dists_list.append(dists_loss)
                id_error_list.append(id_error)
            
            lpips_map[model_name] = lpips_list
            psnr_map[model_name] = psnr_list
            dists_map[model_name] = dists_list
            id_error_map[model_name] = id_error_list

        fig, axs = plt.subplots(2, 2)
        fig.set_figwidth(13)
        fig.set_figheight(10)
        subplot(axs[0, 0], lpips_map, 'LPIPS')
        subplot(axs[0, 1], psnr_map, 'PSNR')
        subplot(axs[1, 0], dists_map, 'DISTS')
        subplot(axs[1, 1], id_error_map, 'ID Error')

        outdir = os.path.join('out', celeb)
        if not os.path.exists(outdir):
                os.makedirs(outdir)
        if pti:
            fig.suptitle('Reconstruction Evaluation with PTI for {}'.format(celeb))    
            plt.savefig(os.path.join(outdir, 'reconstruction_plot_with_pti.png'))
        else:
            fig.suptitle('Reconstruction Evaluation without PTI for {}'.format(celeb))
            plt.savefig(os.path.join(outdir, 'reconstruction_plot_no_pti.png'))

        plt.clf()

def graph_synthesis_results(celeb):
    # first do the graph for all times
    root_folder = os.path.join('synthesized', celeb)
    times = sorted([x for x in os.listdir(root_folder) if x != 'all'])
    models = os.listdir(os.path.join(root_folder, times[0]))
    errors = {}
    for t in times:
        for model in models:
            # read the json file
            with open(os.path.join(root_folder, t, model, 'mean_id_error.json'), 'r') as f:
                error = json.load(f)['mean_id_error']
                if model not in errors:
                    errors[model] = []
                errors[model].append(error)

    for model in models:
        plt.plot(range(len(times)), errors[model], '-o', label=model)
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
            
def graph_interpolation_results(celeb):
    # first do the graph for all times
    root_folder = os.path.join('interpolated', celeb)
    times = sorted([x for x in os.listdir(root_folder) if x != 'all'])
    models = os.listdir(os.path.join(root_folder, times[0]))
    errors = {}
    for t in times:
        for model in models:
            # read the json file
            with open(os.path.join(root_folder, t, model, 'mean_id_error.json'), 'r') as f:
                error = json.load(f)['mean_id_error']
                if model not in errors:
                    errors[model] = []
                errors[model].append(error)

    for model in models:
        plt.plot(range(len(times)), errors[model], '-o', label=model)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('mean ID error (multiview)')
    plt.title('Interpolation Evaluation Results for {}'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'interpolation_plot_over_time.png'))
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
    plt.title('Interpolation Evaluation Results for {} (using all anchors)'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'interpolation_plot_all_anchors.png'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, required=True)
    args = parser.parse_args()
    celeb = args.celeb

    graph_reconstruction_results(celeb)
    graph_synthesis_results(celeb)
    graph_interpolation_results(celeb)
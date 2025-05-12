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

def subplot(ax, map, metric, times):
    colors = plt.cm.viridis([x/len(map.keys()) for x in range(len(map.keys()))])
    for i, model in enumerate(map):
        length = len(map[model])
        ax.plot(range(length), map[model], '-o', label=model, color=colors[i])
    ax.set_xticks(range(length), times)
    ax.set(xlabel='time', ylabel=metric)
    ax.set_title(metric)
    ax.legend()

def subplot_avg(ax, map, metric, times):
    colors = plt.cm.viridis([x/len(map.keys()) for x in range(len(map.keys()))])
    models = list(map.keys())
    average_vals = []
    for i, model in enumerate(models):
        length = len(map[model])
        average_val = sum(map[model]) / len(map[model])
        average_vals.append(average_val)
    ax.bar(models, average_vals, color=colors)

    for i in range(len(models)):
        ax.text(i, average_vals[i], average_vals[i], ha='center')
    #ax.set_xticks(range(length), times)
    #ax.set(xlabel='time', ylabel=metric)
    ax.set_title(metric)
    #ax.legend()

def graph_reconstruction_results(celeb, models_list=None):
    root_folder = os.path.join('embeddings', celeb)
    if not models_list:
        models_list = os.listdir(root_folder)
    models_list = sorted(models_list)
    lpips = LPIPS(net='alex').to(device).eval()
    dists = DISTS().to(device)
    person_identifier = id_utils.PersonIdentifier(id_model, None, None)

    for pti in [False]:#[True, False]:
        lpips_map, psnr_map, dists_map, id_error_map = {}, {}, {}, {}
        for model_name in models_list:
            lpips_list, psnr_list, dists_list, id_error_list = [], [], [], []
            times = sorted([x for x in os.listdir(os.path.join(root_folder, model_name)) if not x.endswith('.json')])
            times = sorted(times, key=lambda x: (x[0].isdigit(), int(x)))
            times = [str(x) for x in range(10)]
            print('times:', times)
            #times = range(10)
            for t in times:
                embedding_folder = os.path.join(root_folder, model_name, '9', str(t))
                if model_name == 'upper_bound':
                    embedding_folder = os.path.join(root_folder, model_name, 'all', str(t))
                print('celeb: {}, model: {}, time: {}'.format(celeb, model_name, t))
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
        subplot(axs[0, 0], lpips_map, 'LPIPS', times)
        subplot(axs[0, 1], psnr_map, 'PSNR', times)
        subplot(axs[1, 0], dists_map, 'DISTS', times)
        subplot(axs[1, 1], id_error_map, 'ID Error', times)

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

def graph_synthesis_results(celeb, models=None):
    # first do the graph for all times
    root_folder = os.path.join('synthesized', celeb)
    if not models:
        models = os.listdir(root_folder)
    models = sorted(models)
    times = sorted([x for x in os.listdir(os.path.join(root_folder, models[0])) if not x.endswith('.json')])
    times = sorted(times, key=lambda x: (x[0].isdigit(), int(x)))
    times = [str(x) for x in range(10)]
    print('times:', times)
    #times = sorted([x for x in os.listdir(os.path.join(root_folder, models[0])) if x != 'all'])
    id_sims = {}
    fids = {}
    for model in models:
        for t in times:
            # get mean ID sim and FID score
            folder = 'all' if model == 'upper_bound' else '9'
            
            # read the json file
            with open(os.path.join(root_folder, model, folder, t, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
                sim = metrics['id_sim']
                fid = metrics['fid']
                if model not in id_sims:
                    id_sims[model] = []
                    fids[model] = []
                id_sims[model].append(sim)
                fids[model].append(fid)

    # get n colors where n is the number of models
    fig, axs = plt.subplots(2, 1)
    fig.set_figwidth(13)
    fig.set_figheight(10)
    subplot(axs[0], id_sims, 'ID Similarity', times)
    subplot(axs[1], fids, 'FID score', times)
    #plt.title('Synthesis Evaluation Results for {} (each time separate)'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'synthesis_plot_over_time.png'))
    plt.clf()

    """
    # then make a bar graph comparing the models' performance overall 
    errors = []
    for model in models:
        # open json file
        with open(os.path.join(root_folder, 'all', model, 'fid_score.json'), 'r') as f:
            error = json.load(f)['fid']
            errors.append(error)
    plt.bar(models, errors)
    plt.xlabel('model')
    plt.ylabel('FID score')
    plt.title('Synthesis Evaluation Results for {} (using all anchors)'.format(celeb))
    plt.savefig(os.path.join('out', celeb, 'synthesis_plot_all_anchors.png'))
    plt.clf()
    """

# old version -- now we are using FID score instead of ID error
def graph_synthesis_results_id_error(celeb):
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
    plt.clf()
            
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
    plt.clf()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--celeb', type=str, required=True)
    parser.add_argument('--models', type=str, required=False)
    args = parser.parse_args()
    celeb = args.celeb

    graph_reconstruction_results(celeb, models_list=args.models.split(',') if args.models else None)
    graph_synthesis_results(celeb, models=args.models.split(',') if args.models else None)
    #graph_interpolation_results(celeb)
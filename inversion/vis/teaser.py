import torch
import cv2
import numpy as np
import os, sys
import shutil
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from camera_utils import LookAtPoseSampler
import dnnlib
import legacy

font_path = 'fonts/LinLibertine_R.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

def subplot(ax, map, metric, models, labels, arrow='', eval='EVALUATION TYPE'):
    #colors = plt.cm.viridis([x/len(map.keys()) for x in range(len(map.keys()))])
    colors = [(r/255, g/255, b/255) for r, g, b in [(238, 102,119), (204,187,68), (68,119,170)]]
    for i, model in enumerate(models):
        length = len(map[model])
        ax.plot(range(length), map[model], '-o', label=model, color=colors[i])
    ax.set_xticks(range(0, 10))
    ax.set_xticklabels(range(1, 11))
    ax.set(xlabel='time', ylabel=metric)
    ax.set_title(f'{eval} {metric}' + arrow)
    #ax.legend()

def graph_results(models_list):
    lpips_map, psnr_map, dists_map, id_sim_map, fid_map, synthesis_id_sim_map = {}, {}, {}, {}, {}, {}
    """
    format of each map should be like this:
    lpips_map = {
        'lower_bound' : {
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        },
        'random' : { 
            'mean' : [0.1, 0.2, 0.3, 0.4, 0.5],
            'std' : [0.01, 0.02, 0.03, 0.04, 0.05]
        }
    }
    """
    celebs = ['Margot']
    for celeb in celebs:
        for model in models_list:
            lpips_list, psnr_list, dists_list, id_sim_list, fid_list, synthesis_id_sim_list = [], [], [], [], [], []
            for t in range(10):
                # get average lpips, psnr, dists, id_error for this model and test cluster
                folder = 'all' if model == 'upper_bound' else '9'
                reconstructions_folder = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings',
                                                        celeb, model, folder, str(t))                                                
                avg_lpips, avg_psnr, avg_dists, avg_id_sim = 0, 0, 0, 0
                    
                for img in range(10):
                    metrics_path = os.path.join(reconstructions_folder, str(img), 'metrics.json')
                    assert os.path.exists(metrics_path), f'{metrics_path} does not exist'

                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    avg_lpips += metrics['lpips']
                    avg_psnr += metrics['psnr']
                    avg_dists += metrics['dists']
                    avg_id_sim += metrics['id_sim']
                avg_lpips /= 10
                avg_psnr /= 10
                avg_dists /= 10
                avg_id_sim /= 10

                lpips_list.append(avg_lpips)
                psnr_list.append(avg_psnr)
                dists_list.append(avg_dists)
                id_sim_list.append(avg_id_sim)

                synthesis_folder = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized',
                                                celeb, model, folder, str(t))
                synthesis_metrics_path = os.path.join(synthesis_folder, 'metrics.json')
                assert os.path.exists(synthesis_metrics_path), f'{synthesis_metrics_path} does not exist'

                with open(synthesis_metrics_path, 'r') as f:
                    synthesis_metrics = json.load(f)
                fid_list.append(synthesis_metrics['fid'])
                if 'id_sim' not in metrics:
                    id_sim = 1 - metrics['mean_id_error']
                else:
                    id_sim = metrics['id_sim']
                synthesis_id_sim_list.append(id_sim)
            
            lpips_map[model] = lpips_list
            psnr_map[model] = psnr_list
            dists_map[model] = dists_list
            id_sim_map[model] = id_sim_list
            fid_map[model] = fid_list
            synthesis_id_sim_map[model] = synthesis_id_sim_list

        labels = ['Lower bound', 'ER-Hull (Ours)', 'Upper bound']
        fig, axs = plt.subplots(2, 1)
        #tight layout
        fig.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.08)
        fig.set_figwidth(5)
        fig.set_figheight(7)
        #increase distance between subplots but not between subplots and the figure
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        subplot(axs[0], lpips_map, 'LPIPS', models_list, labels, arrow=r'$\downarrow$', eval='Reconstruction')
        subplot(axs[1], id_sim_map, 'ID Similarity', models_list, labels, arrow=r'$\uparrow$', eval='Reconstruction')
        #subplot(axs[0], fid_map, 'FID', models_list, labels, arrow=r'$\downarrow$', eval='Synthesis')
        #subplot(axs[1], synthesis_id_sim_map, 'ID Similarity', models_list, labels, arrow=r'$\uparrow$', eval='Synthesis')

        colors = [(r/255, g/255, b/255) for r, g, b in [(238, 102,119), (204,187,68), (68,119,170)]]
        legend_elements = [Line2D([0], [0], marker='o', color=color, label=label, markerfacecolor=color, markersize=10, linestyle='-') 
                        for label, color in zip(labels, colors)]

        # Adjust the right margin of the subplots to create space for the legend
        #plt.subplots_adjust(right=0.8)

        # Add the legend to the figure
        fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.94, 0.8), title='Models', title_fontsize='large', prop=prop)# Add the legend to the figure

        fig.suptitle(f'3D Evaluation After Training On All Clusters For Margot Robbie')
        plt.savefig(os.path.join(f'/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/teaser/graphs_{celeb}.png'), dpi=300)

        plt.clf()

def generate_images(G, img_folder, save_dir, device):
    w = torch.load(os.path.join(img_folder, 'w_optimized.pt')).to(device)

    angle_p = -0.2
    angles_y = np.linspace(-.4, .4, 10)
    angles = [(angle_y, angle_p) for angle_y in angles_y]
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    for j, (angle_y, angle_p) in enumerate(angles):
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        try:
            img = G.synthesis(w, camera_params, noise_mode='const')['image'].detach().cpu()[0]
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).numpy().astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # save novel view image
            img_path = os.path.join(save_dir, f'novel_view_{j}.jpg')
            cv2.imwrite(img_path, img)
        except Exception as e:
            print(f'Error: {e}')
            exit()

def make_novel_views():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiments = ['lower_bound', 'random', 'ransac', 'upper_bound']
    celebs = ['Margot']
    for celeb in celebs:
        for i, experiment in enumerate(experiments):
            # load the model
            model = 'all' if experiment == 'upper_bound' else '9'
            network_pkl = os.path.join(networks, celeb, experiment, f'{model}.pkl')
            with dnnlib.util.open_url(network_pkl) as f:
                G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

            for test_cluster in range(10):
                for test_img_idx in range(10):
                    img_folder = os.path.join(reconstructions, celeb, experiment, model, str(test_cluster), str(test_img_idx))

                    img_save_dir = os.path.join(save_dir, celeb, experiment, f'timestamp_{str(test_cluster)}', f'img_{str(test_img_idx)}')
                    os.makedirs(img_save_dir, exist_ok=True)
                    generate_images(G, img_folder, img_save_dir, device)
                    
                    input_img_path = os.path.join(img_folder, 'input.png')
                    shutil.copy(input_img_path, img_save_dir)

if __name__ == '__main__':

    reconstructions = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings'
    networks = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks'
    save_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/teaser/novel_views_compressed'
    
    models = ['lower_bound', 'ransac', 'upper_bound']
    #graph_results(models)
    make_novel_views()
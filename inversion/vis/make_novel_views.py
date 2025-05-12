import torch
import cv2
import numpy as np
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from camera_utils import LookAtPoseSampler
import dnnlib
import legacy

def generate_images(G, img_folder):
    w = torch.load(os.path.join(img_folder, 'w_optimized.pt')).to(device)

    angle_p = -0.2
    angles_y = np.linspace(-.4, .4, 10)
    for j, (angle_y, angle_p) in enumerate([(.4, angle_p), (0, angle_p), (-.4, angle_p)]):
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
        camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        img = G.synthesis(w, camera_params, noise_mode='const')['image'].detach().cpu()[0]
        img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # save novel view image
        img_path = os.path.join(img_folder, f'novel_view_{j}.png')
        cv2.imwrite(img_path, img)

if __name__ == '__main__':

    celebs = ['Michael']
    reconstructions = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings'
    networks = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks'
    save_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/all_examples'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0], device=device), radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)

    experiments = ['lower_bound', 'random', 'ransac', 'upper_bound']
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
                    
                    generate_images(G, img_folder)
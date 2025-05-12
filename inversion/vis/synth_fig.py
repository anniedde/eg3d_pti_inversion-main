import cv2
import numpy as np
#import torch
import os

celebs = ['Michael']
synthesized = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/synthesized'
save_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/all_synth_examples'
os.makedirs(save_dir, exist_ok=True)

for celeb in celebs:
    for test_cluster in range(10):
        for synth_img_idx in range(20):
            # create 1x4 grid of 512 x 512 images with input and reconstructed images
            grid = np.zeros((512*1, 512*4, 3), dtype=np.uint8)
            
            experiments = ['lower_bound', 'random', 'ransac', 'upper_bound']
            for i, experiment in enumerate(experiments):
                model = 'all' if experiment == 'upper_bound' else '9'
                img_path = os.path.join(synthesized, celeb, experiment, model, str(test_cluster), 'images', f'{synth_img_idx}.png')
                img = cv2.imread(img_path)

                # place input on top of recon_img in the i-th column of the grid
                col_start, col_end = i*512, (i+1)*512
                grid[:, col_start:col_end] = img
            
            save_path = f'{save_dir}/{celeb}_{test_cluster}_{synth_img_idx}.png'
            cv2.imwrite(save_path, grid)


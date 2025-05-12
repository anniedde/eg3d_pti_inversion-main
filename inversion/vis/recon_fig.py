import cv2
import numpy as np
#import torch
import os

celebs = ['IU', 'Harry', 'Margot', 'Michael']
reconstructions = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings'
save_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/all_recon_examples'
os.makedirs(save_dir, exist_ok=True)
for celeb in celebs:
    for test_cluster in range(10):
        for test_img_idx in range(10):
            # create 1x5 grid of 512 x 512 images with input and reconstructed images
            grid = np.zeros((512*1, 512*5, 3), dtype=np.uint8)
            
            experiments = ['lower_bound', 'random', 'ransac', 'upper_bound']
            for i, experiment in enumerate(experiments):
                model = 'all' if experiment == 'upper_bound' else '9'
                img_folder = os.path.join(reconstructions, celeb, experiment, model, str(test_cluster), str(test_img_idx))
                input_img = cv2.imread(os.path.join(img_folder, 'input.png'))
                recon_img = cv2.imread(os.path.join(img_folder, 'before_pti_rgb_proj.png'))
                #novel_img_1 = cv2.imread(os.path.join(img_folder, 'novel_view_0.png'))
                print('img path:', os.path.join(img_folder, 'novel_view_1.png'))
                #novel_img_2 = cv2.imread(os.path.join(img_folder, 'novel_view_1.png'))
                #novel_img_3 = cv2.imread(os.path.join(img_folder, 'novel_view_2.png'))

                # place recon image in the column for this experiment
                grid[0:512, (i+1)*512:(i+2)*512] = recon_img

                
            # put input in first column
            grid[0:512, 0:512] = input_img
            save_path = f'{save_dir}/{celeb}_{test_cluster}_{test_img_idx}.png'
            cv2.imwrite(save_path, grid)


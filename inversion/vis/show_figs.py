import cv2
import os
import numpy as np

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['DISPLAY'] = ':1'

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

examples_folder = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/all_recon_examples'
save_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/vis/supplementary/recon'
os.makedirs(save_dir, exist_ok=True)
celebs = ['Margot', 'Harry', 'IU', 'Michael']
# Naming a window 


for celeb_goal in celebs:
    # make celeb dir
    celeb_dir = os.path.join(save_dir, celeb_goal)
    os.makedirs(celeb_dir, exist_ok=True)
    for test_goal in range(0, 10):
        done = (f'{test_goal}.png' in os.listdir(celeb_dir))
        while not done:
            for file in os.listdir(examples_folder):
                celeb, test_cluster, synth_img_idx = file.split('_')
                test_cluster, synth_img_idx = int(test_cluster), int(synth_img_idx.split('.')[0])
                if test_cluster == test_goal and celeb == celeb_goal:
                    img_path = os.path.join(examples_folder, file)
                    img = cv2.imread(img_path)
                    resize = ResizeWithAspectRatio(img, width=500) # Resize by width OR
                    # resize = ResizeWithAspectRatio(image, height=1280) # Resize by height 
                    #named window
                    cv2.imshow('resize', resize)
                    
                    key = cv2.waitKey(0)
                    # if the user types y, the image will be saved to another folder
                    if key == ord('y'):
                        # save
                        save_path = os.path.join(celeb_dir, f'{test_goal}.png')
                        cv2.imwrite(save_path, img)
                        # move on to the next test_goal
                        done = True
                        break
                    elif key == ord('q'):
                        exit()
                    elif key == ord('n'):
                        continue
                    else:
                        continue
            
    # save grid
    grid = np.zeros((10*512, 5*512, 3), dtype=np.uint8)
    for i in range(10):
        img = cv2.imread(os.path.join(celeb_dir, f'{i}.png'))
        grid[i*512:(i+1)*512, :, :] = img
    cv2.imwrite(os.path.join(save_dir, f'{celeb_goal}_grid.png'), grid)
cv2.destroyAllWindows()
import argparse
import os
import numpy as np
from itertools import combinations
from camera_utils import LookAtPoseSampler
from PIL import Image
from utils_copy import id_utils # make a copy of utils to avoid import error as eg3d also has utils
import torch
import json
import dnnlib
import legacy
from torchvision.utils import make_grid, save_image

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

device=torch.device('cuda')
network_pkl='/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/networks/lower_bound_t4_no_lora.pkl'
results = {}
for t in range(5):
    # get all training samples and their corresponding ws
    ws = []
    training_imgs = []
    folder = '/playpen-nas-ssd/awang/data/luchao_preprocessed'
    ids = []
    image_paths = os.listdir(folder)
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name = fileName.split('_')[0]
            if id_name not in ids:
                ids.append(id_name)
    ids.sort()
    id_name = ids[t]
    out_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/out/lower_bound_' + id_name + '_interpolations'
    if not (os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
    person_identifier = id_utils.PersonIdentifier(id_model, None, None)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    input_features = []
    for fileName in image_paths:
        if fileName.endswith('_latent.npy') and 'mirror' not in fileName:
            if (fileName.split('_')[0] == id_name):
                ws.append(np.load(os.path.join(folder, fileName), allow_pickle=True))
            
        if fileName.endswith('.png') and 'mirror' not in fileName:
            if (fileName.split('_')[0] == id_name):
                img = Image.open(os.path.join(folder, fileName)).convert('RGB')
                img = np.array(img)
                feature = person_identifier.get_feature(img)
                input_features.append(feature)

    pairs = combinations(ws, 2)
    error_list = []
    for pair in pairs:
        w1 = pair[0]
        w2 = pair[1]

        id_errors = []
        #interpolated_images = []
        for alpha in np.linspace(0, 1, 10):
            print(alpha)
            w = alpha * w1 + (1 - alpha) * w2
            sim_list = []
            imgs = []
            for idx in range(1):
                print('idx: ', idx)
                pitch_range = 0.25
                yaw_range = 0.35
                camera_lookat_point = torch.tensor([0, 0, 0], device=device)
                cam2world_pose = LookAtPoseSampler.sample(3.14/2 + yaw_range * np.sin(2 * 3.14 * idx / 10),
                                                        3.14/2 -0.05 + pitch_range * np.cos(2 * 3.14 * idx / 10),
                                                        camera_lookat_point, radius=2.7, device=device)
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1) # (1, 25)
                c = c.type(torch.float32)
                img = G.synthesis(ws=torch.from_numpy(w).to(device), c=c[0:1].to(device), noise_mode='const')['image'][0]
                normalized = np.array(layout_grid(torch.stack([img])))
                feature = person_identifier.get_feature(normalized)
                #img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                #imgs.append(img[0])
                #if (idx==0):
                #    interpolated_images.append(img[0])
                sims = [person_identifier.compute_similarity(feature, v) for v in input_features]
                sim = max(sims).item()
                sim_list.append(sim)
            max_sim = max(sim_list)
            # id score is 1-complement of the distance
            id_errors.append(1-max_sim)
            #grid = make_grid(imgs, nrow=5, normalize=True)
            #Image.fromarray(grid.cpu().numpy(), 'RGB').save('TESTIMAGE-alpha{}.png'.format(alpha))
            #save_image(grid, out_dir + "/grid-alpha{}.jpg".format(alpha))
        id_error_mean = round(np.mean(id_errors), 3)
        error_list.append(id_error_mean)
        #grid = make_grid(interpolated_images, nrow=5, normalize=True)
        #save_image(grid, out_dir + "/grid-idx0.jpg".format(alpha))
    
    results[t] = np.mean(error_list)

with open('lower_bound_interpolation_2.json'.format(t), 'w') as f:
    json.dump(results, f)
        
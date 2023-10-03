import os
import json
from configs import global_config, paths_config, hyperparameters
import torch
from torchvision import transforms
from lpips import LPIPS
from DISTS_pytorch import DISTS
import pickle
import dnnlib
import PIL
from utils_copy import id_utils # make a copy of utils to avoid import error as eg3d also has utils
import legacy
import numpy as np

def denorm(img):
    # img: [b, c, h, w]
    img_to_return = (img + 1) * 127.5
    img_to_return = img_to_return.permute(0, 2, 3, 1).clamp(0, 255)
    return img_to_return # [b, h, w, c]

results = {}

# ---------------------------------- network --------------------------------- #
device=torch.device('cuda')
network_pkl = paths_config.eg3d_ffhq 
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
id_model = '/playpen-nas-ssd/luchao/projects/mystyle/pretrained/model_ir_se50.pth'
person_identifier = id_utils.PersonIdentifier(id_model, None, None)
paths_config_path = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/configs/paths_config.py'

for t in range(1,6):
    folder = '/playpen-nas-ssd/awang/data/luchao_test_10_images/t{}'.format(t)
    image_paths = os.listdir(folder)
    time_result = {}
    for fileName in image_paths:
        if fileName.endswith('.png'):
            id_name_ = fileName # with png extension
            id_name = fileName.split('.')[0]
            model_name = paths_config.eg3d_ffhq.split('/')[-1]
            model_name = os.path.splitext(model_name)[0]
            print('id_name: ', id_name)
            dataset_json = os.path.join(folder, 'epoch_20_000000', 'cameras.json')
            print('dataset json: ', dataset_json)
            embedding_folder = 'luchao_test_t{}'.format(t)
            
            if not os.path.isdir('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/' + embedding_folder): 
                os.system('sed -i "s/input_id = .*/input_id = \'{}\'/g" {}'.format(id_name, paths_config_path))
                os.system('sed -i "s@input_pose_path = .*@input_pose_path = \'{}\'@g" {}'.format(os.path.join(folder, 'epoch_20_000000', 'cameras.json'), paths_config_path))
                os.system('sed -i "s@input_data_path = .*@input_data_path = \'{}\'@g" {}'.format(os.path.join(folder, 'crop_1024'), paths_config_path))
                os.system('sed -i "s/input_data_id = .*/input_data_id = \'{}\'/g" {}'.format(embedding_folder, paths_config_path))
                # run the script
                print('running pti')
                cmd = 'python run_pti.py'
                os.system(cmd)
            

            # ------------------------- reconstruction evaluation ------------------------ #

            # ----------------------------------- lpips ---------------------------------- #
            
            with open(dataset_json, 'r') as f:
                print('dataset json: ', dataset_json)
                dataset = json.load(f)
                print(dataset)
                target_pose = np.asarray(dataset[id_name]['pose']).astype(np.float32)
                f.close()
            o = target_pose[0:3, 3]
            print("norm of origin before normalization:", np.linalg.norm(o))
            o = 2.7 * o / np.linalg.norm(o)
            target_pose[0:3, 3] = o
            target_pose = np.reshape(target_pose, -1)    

            intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
            target_pose = np.concatenate([target_pose, intrinsics])
            target_pose = torch.tensor(target_pose, device=device).unsqueeze(0)
            w_location = os.path.join('/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings', embedding_folder, 'PTI', id_name + '_' + paths_config.eg3d_ffhq.split('/')[-1],'optimized_noise_dict.pickle')

            w_path_dir = f'{paths_config.embedding_base_dir}/{embedding_folder}'
            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{id_name}'
            model_name = paths_config.eg3d_ffhq.split('/')[-1]
            embedding_dir += f'_{model_name}'
            w_location = os.path.join(embedding_dir, 'optimized_noise_dict.pickle')

            print('w_location: ', w_location)
            with open(w_location, 'rb') as f:
                ws = pickle.load(f)['projected_w']
            ws = torch.from_numpy(ws).to(device)
            w_ = ws.squeeze().unsqueeze(0) # torch.Size([1, 14, 512])
            output_img = G.synthesis(ws=w_, c=target_pose, noise_mode='const')['image'].to(device) # [1, 3, 512, 512] [B, C, H, W]
            input_img = PIL.Image.open(os.path.join(os.path.join(folder, 'crop_1024'), f'{id_name}.png')).convert('RGB') # [512, 512, 3] [H, W, C]
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_img = transform(input_img).to(device).unsqueeze(0) # [1, 512, 512, 3] [B, H, W, C]
            
            # lpips requires normalized images
            
            lpips = LPIPS(net='alex').to(device).eval()
            lpips_loss = lpips(input_img, output_img)
            lpips_loss = lpips_loss.squeeze().item()
            lpips_loss = round(lpips_loss, 3)

            # ----------------------------------- psnr ----------------------------------- #
            # psnr requires denormalized images
            input_img_denorm = denorm(input_img)
            output_img_denorm = denorm(output_img)
            # type casting to uint8 then back to float32 is necessary for psnr calculation
            input_img_denorm = input_img_denorm.to(torch.uint8).to(torch.float32)
            output_img_denorm = output_img_denorm.to(torch.uint8).to(torch.float32)
            mse = torch.mean((input_img_denorm - output_img_denorm) ** 2)
            psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
            psnr = psnr.squeeze().item()
            psnr = round(psnr, 3)

            # ----------------------------------- DISTS ---------------------------------- #
            # DISTS requires normalized images
            
            dists = DISTS().to(device)
            dists_loss = dists(input_img, output_img)
            dists_loss = dists_loss.squeeze().item()
            dists_loss = round(dists_loss, 3)

            # --------------------------------- id_error --------------------------------- #
            # # id_error requires denormalized images
            input_img_denorm = denorm(input_img)
            output_img_denorm = denorm(output_img)
            input_feature = person_identifier.get_feature(input_img_denorm.squeeze())
            output_feature = person_identifier.get_feature(output_img_denorm.squeeze())
            sim = person_identifier.compute_similarity(input_feature, output_feature)
            sim = sim.item()
            sim = round(sim, 3)
            id_error = 1 - sim

            im_results = {
                'lpips' : lpips_loss,
                'psnr' : psnr,
                'dists': dists_loss,
                'id_error': id_error
            }
            time_result[id_name] = im_results
    
    results['t{}'.format(t)] = time_result

with open('/playpen-nas-ssd/awang/data/luchao_test_10_images/reconstruction_results_lora.json', 'w') as out:
    json.dump(results, out)

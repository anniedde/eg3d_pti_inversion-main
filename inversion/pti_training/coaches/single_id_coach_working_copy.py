import os
import torch
from tqdm import tqdm
from configs import paths_config, hyperparameters, global_config
from pti_training.coaches.base_coach import BaseCoach
from utils.log_utils import log_images_from_w
from PIL import Image
import imageio
import numpy as np
import pickle
import shutil
import re

class SingleIDCoachWorkingCopy(BaseCoach):

    def __init__(self, data_loader, use_wandb, input_pose_path, input_id, network_path, embedding_folder):
        super().__init__(data_loader, use_wandb, input_pose_path, input_id, network_path)
        self.embedding_folder = embedding_folder

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{self.embedding_folder}'
        os.makedirs(w_path_dir, exist_ok=True)

        use_ball_holder = True

        for idx, (fname, image) in enumerate(tqdm(self.data_loader)):
            image_name = fname[0]

            self.restart_training()

            # ! if self.G contains layers containing lora, freeze them
            #for name, param in self.G.named_parameters():
            #    if 'lora' in name:
            #        param.requires_grad = False

            # REMOVE LATER - TEMPORARY
            ##################################################
            self.G.scaling_factor = torch.nn.Parameter(torch.tensor(1.))
            if 'lora_expansion' in self.network_path:
                rank = int(self.embedding_folder.split('/')[-1]) + 1
                scaling_factor = 1.0 / rank

                for name, module in self.G.named_modules():
                    if 'conv0' in name or 'conv1' in name or 'torgb' in name or 'skip' in name or 'affine' in name:
                        module.scaling = self.G.scaling_factor
            ##################################################

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{image_name}'
            if os.path.exists(embedding_dir):
                shutil.rmtree(embedding_dir)
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots and idx > 0:
                w_pivot = self.load_inversions(w_path_dir, image_name)
            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                import time
                #initial_w = torch.from_numpy(np.load('/playpen-nas-ssd/awang/data/orange_cat_preprocessed/2_latent.npy')).to(global_config.device)
                #initial_w = initial_w[0][0].unsqueeze(0).unsqueeze(0)
                start = time.time()
                w_pivot, noise_bufs, all_w_opt = self.calc_inversions(image, image_name, embedding_dir, 
                write_video=True, initial_w=None) 
                end = time.time()
                print("time to calc inversions:", end - start)
            
            w_pivot = w_pivot.to(global_config.device)
            torch.save(w_pivot, f'{embedding_dir}/w_optimized.pt')

            if 'lora_expansion' in self.network_path:
                for name, module in self.G.named_modules():
                    if 'conv0' in name or 'conv1' in name or 'torgb' in name or 'skip' in name or 'affine' in name:
                        module.scaling = self.G.scaling_factor
                        print('scaling factor for layer ', name, ' is ', module.scaling)
            print('scaling factor for G is ', self.G.scaling_factor)

            # Save optimized noise.
            """
            for noise_buf in noise_bufs:
                noise_bufs[noise_buf] = noise_bufs[noise_buf].detach().cpu().numpy()
            optimized_dict = {
                'projected_w': w_pivot.detach().cpu().numpy(),
                'all_w_opt': all_w_opt.cpu().numpy(),
                'noise_bufs': noise_bufs
            }
            with open(f'{embedding_dir}/optimized_noise_dict.pickle', 'wb') as handle:
                pickle.dump(optimized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            
            # save w_opt for gen_videos.py
            out_path = paths_config.checkpoints_dir
            # remove file if exists
            if os.path.exists(f'{out_path}/w_pivot.pkl'):
                os.remove(f'{out_path}/w_pivot.pkl')
            with open(f'{out_path}/w_pivot.pkl', 'wb') as handle:
                pickle.dump(w_pivot.detach().cpu().numpy(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            real_image = 0.5 * (image + 1) * 255
            real_image = real_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

            vid_path = f'{embedding_dir}' + '/' + 'final_rgb_proj.mp4'
            rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

            # zero out irrelevant lora weights
            for name, param in self.G.named_parameters():
                rank = int(self.embedding_folder.split('/')[-1]) + 1
                if 'lora' in name:
                    match = re.search(r'rank_index_(\d+)', name)
                    if match:
                        index = int(match.group(1))
                        if index > rank:
                            param.data = torch.zeros_like(param.data)
                            print('zeroing weight')

            np.random.seed(1989)
            torch.manual_seed(1989)
            start = time.time()
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot)['image']

                loss, _, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name, self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                # if loss_lpips <= hyperparameters.LPIPS_value_threshold:
                #     break
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                if i % 5 == 0:
                    synth_image = generated_images
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    rgb_video.append_data(np.concatenate([real_image, synth_image], axis=1))

                if i == hyperparameters.max_pti_steps - 1:
                    synth_image = generated_images
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    Image.fromarray(synth_image, 'RGB').save(f'{embedding_dir}' + '/' + 'final_rgb_proj.png')
                    Image.fromarray(real_image, 'RGB').save(f'{embedding_dir}' + '/' + 'input.png')

                global_config.training_step += 1
                log_images_counter += 1
            end = time.time()
            print("time to update model:", end - start)
            self.image_counter += 1
            rgb_video.close()

            # torch.save(self.G, f'{embedding_dir}/model_{image_name}.pt')
            
            # save model for gen_videos.py
            torch.save(self.G.state_dict(), f'{embedding_dir}/tuned_G.pt')
            out_path = paths_config.checkpoints_dir
            # remove file if exists
            if os.path.exists(f'{out_path}/tuned_G.pt'):
                os.remove(f'{out_path}/tuned_G.pt')
            torch.save(self.G.state_dict(), f'{out_path}/tuned_G.pt')
            
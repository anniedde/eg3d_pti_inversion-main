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


class SingleIDCoach(BaseCoach):

    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)

    def train(self):

        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)

        use_ball_holder = True

        for idx, (fname, image) in enumerate(tqdm(self.data_loader)):
            image_name = fname[0]
            print("image name:", image_name)

            self.restart_training()

            if self.image_counter >= hyperparameters.max_images_to_invert:
                break

            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            os.makedirs(embedding_dir, exist_ok=True)

            w_pivot = None

            if hyperparameters.use_last_w_pivots and idx > 0:
                w_pivot = self.load_inversions(w_path_dir, image_name)
            elif not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot, noise_bufs, all_w_opt = self.calc_inversions(image, image_name, embedding_dir, write_video=True)

            w_pivot = w_pivot.to(global_config.device)

            # Save optimized noise.
            for noise_buf in noise_bufs:
                noise_bufs[noise_buf] = noise_bufs[noise_buf].detach().cpu().numpy()
            optimized_dict = {
                'projected_w': w_pivot.detach().cpu().numpy(),
                'all_w_opt': all_w_opt.cpu().numpy(),
                'noise_bufs': noise_bufs
            }
            with open(f'{embedding_dir}/optimized_noise_dict.pickle', 'wb') as handle:
                pickle.dump(optimized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            log_images_counter = 0
            real_images_batch = image.to(global_config.device)

            real_image = 0.5 * (image + 1) * 255
            real_image = real_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            """
            vid_path = f'{embedding_dir}' + '/' + 'final_rgb_proj.mp4'
            rgb_video = imageio.get_writer(vid_path, mode='I', fps=10, codec='libx264', bitrate='16M')

            np.random.seed(1989)
            torch.manual_seed(1989)
            for i in tqdm(range(hyperparameters.max_pti_steps)):
                generated_images = self.forward(w_pivot)['image']

                loss, _, _ = self.calc_loss(generated_images, real_images_batch, image_name, self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
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

                global_config.training_step += 1
                log_images_counter += 1

            self.image_counter += 1
            rgb_video.close()

            torch.save(self.G, f'{embedding_dir}/model_{image_name}.pt')
            """

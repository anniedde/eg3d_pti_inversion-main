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

class MultiIDCoach(BaseCoach):
    
    def __init__(self, data_loader, use_wandb):
        super().__init__(data_loader, use_wandb)
        
    def train(self):
        
        w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
        os.makedirs(w_path_dir, exist_ok=True)
        os.makedirs(f'{w_path_dir}/{paths_config.pti_results_keyword}', exist_ok=True)
        
        use_ball_holder = True
        w_pivots = []
        images = []

        for fname, image in self.data_loader:
            if self.image_counter >= hyperparameters.max_images_to_invert:
                break
            
            image_name = fname[0]
            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            if os.path.exists(embedding_dir):
                shutil.rmtree(embedding_dir)
            os.makedirs(embedding_dir, exist_ok=True)
            w_pivot = None
            
            if not hyperparameters.use_last_w_pivots or w_pivot is None:
                w_pivot, noise_bufs, all_w_opt = self.calc_inversions(image, image_name, embedding_dir, write_video=True)
                pass
            else:
                raise NotImplementedError()
            
            w_pivot = w_pivot.to(global_config.device)
            w_pivots.append(w_pivot)
            images.append((image_name, image))
            self.image_counter += 1
            out_path = paths_config.multi_id_output_dir
            # remove file if exists
            fn = '{out_path}/w_pivot_{}.pkl'.format(image_name)
            if os.path.exists(fn):
                os.remove(fn)
            with open(fn, 'wb') as handle:
                pickle.dump(optimized_dict['projected_w'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                pass
            pass
        
        np.random.seed(1989)
        torch.manual_seed(1989)
        for i in tqdm(range(hyperparameters.max_pti_steps)):
            self.image_counter = 0
            
            for data, w_pivot in zip(images, w_pivots):
                image_name, image = data
                
                if self.image_counter >= hyperparameters.max_images_to_invert:
                    break
                
                real_images_batch = image.to(global_config.device)

                generated_images = self.forward(w_pivot)['image']
                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, image_name,
                                      self.G, use_ball_holder, w_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = global_config.training_step % hyperparameters.locality_regularization_interval == 0

                global_config.training_step += 1
                self.image_counter += 1

                pass
            pass
        
        out_path = paths_config.multi_id_output_dir
        # remove file if exists
        if os.path.exists(f'{out_path}/tuned_multi_G.pt'):
            os.remove(f'{out_path}/tuned_multi_G.pt')
        torch.save(self.G.state_dict(), f'{out_path}/tuned_multi_G.pt')
        pass
    pass

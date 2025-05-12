import argparse
import os
from PIL import Image
import torch
from torchvision.utils import make_grid, save_image
from torchvision import transforms

transform = transforms.Compose([
                    transforms.ToTensor()
                ])

def compile_synthesis_images(celeb, model_list, out):
    synthesis_dir = os.path.join('synthesized', celeb)
    for year in [x for x in os.listdir(synthesis_dir) if x.isdigit()]:
        out_folder = os.path.join(out, year)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        img_list = []
        for i in range(5):
            # compile the images
            for model in model_list:
                img_loc = os.path.join(synthesis_dir, year, model, 'images', f'{i}.png')
                img = transform(Image.open(img_loc).convert('RGB'))
                img_list.append(img)

        grid = make_grid(img_list, normalize=True, nrow=len(model_list))
        save_image(grid, os.path.join(out_folder, f'grid.png'))

def compile_reconstruction_images(celeb, model_list, out):
    data_loc = f'/playpen-nas-ssd/awang/data/{celeb}'
    if 'Margot_up_to_5' in celeb:
        data_loc = f'/playpen-nas-ssd/awang/data/Margot_up_to_5'
    for year in [x for x in os.listdir(data_loc) if x.isdigit()]:
        out_folder = os.path.join(out, year)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        test_data_loc = os.path.join(data_loc, year, 'test', 'preprocessed')
        grid_image_list = []
        for img_name in os.listdir(test_data_loc):
            if os.path.isdir(os.path.join(test_data_loc, img_name)) and "mirror" not in img_name:
                # compile the images
                img_list = []
                # first append input test image
                img = transform(Image.open(os.path.join(test_data_loc, f'{img_name}.png')).convert('RGB'))
                img_list.append(img)
                for model in model_list:
                    img_loc = os.path.join('embeddings', celeb, model, year, img_name, 'before_pti_rgb_proj.png')
                    img = transform(Image.open(img_loc).convert('RGB'))
                    img_list.append(img)

                grid = make_grid(img_list, normalize=True)
                save_image(grid, os.path.join(out_folder, f'{img_name}.png'))
                grid_image_list += img_list

        grid = make_grid(grid_image_list, normalize=True, nrow=len(model_list)+1)
        save_image(grid, os.path.join(out_folder, f'grid.png'))

parser = argparse.ArgumentParser()
parser.add_argument("--celeb", help="Name of the celebrity")
parser.add_argument("--models", help="List of models")
parser.add_argument("--name", help="descriptor")
args = parser.parse_args()

reconstruction_out_folder = os.path.join('out', args.celeb, f'compiled_reconstructions_{args.name}' if args.name else 'compiled_reconstructions')
if not os.path.exists(reconstruction_out_folder):
    os.makedirs(reconstruction_out_folder)
synthesis_out_folder = os.path.join('out', args.celeb, f'compiled_synthesis_{args.name}' if args.name else 'compiled_synthesis')
if not os.path.exists(synthesis_out_folder):
    os.makedirs(synthesis_out_folder)

models = args.models.split(',')
compile_reconstruction_images(args.celeb, models, reconstruction_out_folder)
compile_synthesis_images(args.celeb, models, synthesis_out_folder)

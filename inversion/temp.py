import os
from PIL import Image
import numpy as np

imgs_dir = '/playpen-nas-ssd/awang/eg3d_pti_inversion-main/inversion/embeddings/orange_cat_original/upper_bound_no_lora'

def stack_imgs(id_name):
    personalized = Image.open(f'out/cat_renderings/interpolate_video_{id_name}_personalized_no_pti_geometry.png')
    pretrained = Image.open(f'out/cat_renderings/interpolate_video_{id_name}_pretrained_geometry.png')
    imgs = [pretrained, personalized]
    imgs_comb = np.vstack(imgs)
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(f'out/cat_renderings/{id_name}_no_pti.png' )

def render(id_name):
    cmd = f'CUDA_VISIBLE_DEVICES=1 python render_geometry.py --fname=/playpen-nas-ssd/awang/my3dgen/eg3d/output/{id_name}_no_pti.npy --sigma-threshold=10 --outdir=out/cat_renderings --id_name={id_name} --model_name=personalized_no_pti'
    os.system(cmd)
    #cmd = f'CUDA_VISIBLE_DEVICES=1 python render_geometry.py --fname=/playpen-nas-ssd/awang/my3dgen/eg3d/output/{id_name}_pretrained.npy --sigma-threshold=10 --outdir=out/cat_renderings --id_name={id_name} --model_name=pretrained_pti'
    #os.system(cmd)
    
for file in os.listdir(imgs_dir):
    id_name = file.split('.')[0]
    render(id_name)
    stack_imgs(id_name)
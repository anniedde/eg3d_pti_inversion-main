# test the performance of the given model on all test images
# ! change paths_config.py, image_dir, `paths_config` variable to your own path and dataset_json in gen_videos.py

from configs import paths_config
import os

image_dir = '{}/eg3d/dataset/{}_test'.format(paths_config.PATH_PREFIX, paths_config.run_name)
dest_path = paths_config.input_data_path # Where the input image resides
os.makedirs(dest_path, exist_ok=True)
# save the model
model_path = paths_config.eg3d_ffhq
os.system('cp {} {}'.format(model_path, './out/'))
# create a readme with model_path
with open('./out/README.txt', 'w') as f:
    f.write('model_path: {}'.format(model_path))

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
# * ignore mirrored images in test images
image_files = [f for f in image_files if f.endswith('.png') and 'mirror' not in f]

if not os.path.exists(dest_path):
    os.makedirs(dest_path)
for image_file in sorted(image_files):
    try:
        # clear the dest_path
        for f in os.listdir(dest_path):
            os.remove(os.path.join(dest_path, f))
        # copy the image to dest_path
        os.system('cp {} {}'.format(image_file, dest_path))
        # change the input_id in paths_config.py
        paths_config_path = '{}/eg3d_pti_inversion/inversion/configs/paths_config.py'.format(paths_config.PATH_PREFIX)
        image_name = os.path.basename(image_file)
        os.system('sed -i "s/input_id = .*/input_id = \'{}\'/g" {}'.format(os.path.basename(image_name), paths_config_path))
        # run the script
        print('running pti')
        cmd = 'python run_pti.py'
        os.system(cmd)
        print('outputing rgb videos')
        cmd = 'python gen_videos.py --outdir=./out --trunc=0.7 --seeds=0 --reload_modules=True --nrr=128' # use 128 to align with the pretrained model
        os.system(cmd)
        # cmd = 'python gen_videos.py --outdir=out --trunc=0.7 --seeds=0 --reload_modules=True --image_mode=image_depth --nrr=128'
        # os.system(cmd)
        # cmd = 'python gen_videos.py --outdir=out --trunc=0.7 --seeds=0 --reload_modules=True --image_mode=image_raw --nrr=128'
        # os.system(cmd)

        # geometry
        print('outputing geometry')
        cmd = 'python gen_samples.py --outdir=./out --trunc=0.7 --seeds=0 --reload_modules=True --shapes=True' # get npy meshes
        os.system(cmd)
        #cmd = 'python render_geometry.py --fname ./out/seed0000.npy --sigma-threshold 10 --outdir ./out' # render meshes
        #os.system(cmd)

    # keyboard interrupt
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        # exit
        exit()
    # other exception
    except Exception as e:
        print(e)

# ------------------------ finally wrap up the output ------------------------ #
# move the output files into a folder
run_name = paths_config.run_name.lower()
model_name = paths_config.eg3d_ffhq.split('/')[-1].split('.')[0]
dest_dir = './out/{}_{}'.format(run_name, model_name)
files = os.listdir('./out')
files = [f for f in files if os.path.isfile(os.path.join('./out', f))]
os.makedirs(dest_dir, exist_ok=True)
for f in files:
    os.system('mv ./out/{} {}'.format(f, dest_dir))
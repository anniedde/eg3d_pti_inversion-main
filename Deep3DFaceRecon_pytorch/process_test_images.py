"""
    Processes a directory containing *.jpg/png and outputs crops and poses.
"""
import glob
import os
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/media/data6/ericryanchan/mafu/Deep3DFaceRecon_pytorch/test_images')
parser.add_argument('--gpu', default=0)
args = parser.parse_args()

print('Processing images:', sorted(glob.glob(os.path.join(args.input_dir, "*"))))

# Compute facial landmarks.
print("Computing facial landmarks for model...")
cmd = "python batch_mtcnn.py"
input_flag = " --in_root " + args.input_dir
cmd += input_flag
subprocess.run([cmd], shell=True)

# Run model inference to produce crops and raw poses.
print("Running model inference...")
cmd = "python test.py"
input_flag = " --img_folder=" + args.input_dir
gpu_flag = " --gpu_ids=" + str(args.gpu) 
model_name_flag = " --name=pretrained"
model_file_flag = " --epoch=20 "
cmd += input_flag + gpu_flag + model_name_flag + model_file_flag
subprocess.run([cmd], shell=True)

# Process poses into our representation -- produces a cameras.json file.
print("Processing final poses...")
cmd = "python 3dface2idr.py"
input_flag = " --in_root " + os.path.join(args.input_dir, "epoch_20_000000")
cmd += input_flag
subprocess.run([cmd], shell=True)

# Perform final cropping of 1024x1024 images.
print("Processing final crops...")
cmd = "python final_crop.py"
input_flag = " --in_root " + os.path.join(args.input_dir, "crop_1024")
input_flag_2 = " --in_root_2 " + os.path.join(args.input_dir)
cmd += input_flag
cmd += input_flag_2
subprocess.run([cmd], shell=True)

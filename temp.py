import gdown
import argparse
import os
import shutil

"""
folder = '/playpen-nas-ssd/awang/data/Barack_test_easy'

image_paths = os.listdir(folder)
for fileName in image_paths:
    if fileName.endswith('.png'):
        id_name = fileName.split('.')[0]
        image_folder = os.path.join(folder, id_name)
        os.mkdir(image_folder)
        shutil.copy(os.path.join(folder, 'crop_1024', fileName), os.path.join(image_folder, fileName))
        #shutil.copy(os.path.join(folder, fileName), os.path.join(image_folder, fileName))
"""
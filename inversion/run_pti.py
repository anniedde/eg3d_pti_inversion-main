import sys
sys.path.append("../")

from torch.utils.tensorboard import SummaryWriter
import torch
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config, hyperparameters
import shutil
import traceback

from pti_training.coaches.single_id_coach import SingleIDCoach
from pti_training.coaches.single_id_coach_grayscale import SingleIDCoachGrayscale
from utils.ImagesDataset import ImagesDataset, GrayscaleImagesDataset, DECADataset
#from utils.parse_args import parse_args

def run_PTI_func(input_id, input_pose_path, input_data_path, embedding_folder, eg3d_ffhq):
    #parse_args()
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    dataset = ImagesDataset(input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToTensor()]))
    try:
        dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=False)
        coach = SingleIDCoach(dataloader, False, input_pose_path, input_id, eg3d_ffhq, embedding_folder)
    
        coach.train()
    except Exception as e:
        print('error: ', e)
        traceback.print_exc()
        exit()
    return global_config.run_name

def run_PTI_func_image(input_id, input_pose_path, input_data_path, embedding_folder, eg3d_ffhq, rank):
    #parse_args()
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    dataset = ImagesDataset(input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToTensor()]))
    try:
        dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=False, )
        coach = SingleIDCoach(dataloader, False, input_pose_path, input_id, eg3d_ffhq, embedding_folder)
    
        coach.train()
    except Exception as e:
        print('error: ', e)
        traceback.print_exc()
        exit()
    return global_config.run_name
"""
def run_PTI(run_name='', use_wandb=False):
    #parse_args()
    #os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1
    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToTensor()]))

    dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=False)
    coach = SingleIDCoach(dataloader, use_wandb)

    coach.train()
    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='')
"""
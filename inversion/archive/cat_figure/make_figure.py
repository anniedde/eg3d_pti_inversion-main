import os
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import torch
from lpips import LPIPS
from DISTS_pytorch import DISTS
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import json
import numpy as np
import cv2

# Read the image
left = cv2.imread('2-left.png')
right = cv2.imread('2-right.png')
print(left.shape)
height = left.shape[0]
width = left.shape[1]

# Cut the image in half
width_cutoff = width // 2
new_image = np.empty_like(left)
new_image[:, :width_cutoff] = left[:, :width_cutoff]
new_image[:, width_cutoff:] = right[:, width_cutoff:]

cv2.imwrite('2.png', new_image)
from torchvision.io import read_image
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms
from torchvision import torch

import os

tensors = []

transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float),
])

imgs = [1, 2, 3, 5, 6, 7]
#for file in sorted(os.listdir('.')):
for num in imgs:
    file = f'{num}.png'
    if file.endswith('.png'):
        transformed_tensor = transform(read_image(file))
        tensors.append(transformed_tensor)

grid = make_grid(tensors, nrow=3, padding=0)

save_image(grid, "grid.png")
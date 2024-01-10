import os
from PIL import Image
import numpy as np
import subprocess

pairs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
num_processes = 3

chunks = [pairs[i:i + num_processes] for i in range(0, len(pairs), num_processes)]
print(chunks)
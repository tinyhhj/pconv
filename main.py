import argparse
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--mask' , type=str, default='irregular_mask/disocclusion_img_mask')
parser.add_argument('--batch_size',type=int ,default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_iter',type=int, default=100000)
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--checkpoint',type=str, default='checkpoint')
parser.add_argument('--iter_log', type=int, default=1000)
parser.add_argument('--iter_save',type=int, default=1000)
parser.add_argument('--iter_sample',type=int, default=1000)
parser.add_argument('--iter_eval', type=int, default=1000)

args = parser.parse_args()

# make dir data checkpoint
os.makedirs(args.data,exist_ok=True)
os.makedirs(args.checkpoint,exist_ok=True)
# if we do sample make dir
if args.iter_sample > 0:
    os.makedirs(os.path.join(args.checkpoint,'sample'),exist_ok=True)





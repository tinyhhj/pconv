import torch
from blocks import GatedConv2d

class EUnet(torch.nn.Module):
    def __init__(self):
        blocks = []
        blocks.append(GatedConv2d())
    def forward(self,x):
        pass
# layernorm
# pointwise conv
    # expansion factor of 2
# glu activation
# 1d depthwise conv
# batch norm
# swish activation
# pointwise conv
# dropout
# + (x)
import torch
import torch.nn as nn

class ConvolutionModule():
    def __init__(self):
        self.layernorm = nn.LayerNorm(256)
        self.pointwise = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(1,1))

# spech aug
# conv subsamplin
# linear
# dropout
# conformer block x N
    # ff (1/2)x
    # +
    # mhsa_module
    # +
    # convolution_module
    # +
    # ff (1/2)x
    # +
    # layernorm


import torch
import torch.nn as nn
from model.ff_module import FeedForwardModule
from model.mhsa_module import MultiHeadedAttention
from model.convolution_module import ConvolutionModule

class ConformerBlock(nn.Module):
    def __init__(self, dim_model, attn_heads, kernel_size):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForwardModule(d_module=dim_model, expansion_factor=4)
        self.mhsa = MultiHeadedAttention(attn_heads, dim_model)
        self.conv = ConvolutionModule(d_module=dim_model, expansion_factor=2, kernel_size=kernel_size)
        self.ff2 = FeedForwardModule(d_module=dim_model, expansion_factor=4)
        self.layernorm = nn.LayerNorm(dim_model)
    
    def forward(self, x):
        out = self.ff1(x)
        out += x
        out_prev = out
        out = self.mhsa(out, out, out)
        out += out_prev
        out_prev = out
        out = self.conv(out)
        out += out_prev
        out_prev = out
        out = self.ff2(out)
        out += out_prev
        out_prev = out
        out = self.layernorm(out)
        return out

class ConformerEncoder(nn.Module):
    # The input_dim: "We extracted 80-channel filterbanks features 
    # computed from a 25ms window with a stride of 10ms."
    def __init__(self, encoder_dim=256, num_blocks=16, att_heads=4, kernel_size=31, input_dim=80):
        super(ConformerEncoder, self).__init__()
        # conv sub sampling
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=encoder_dim, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=encoder_dim,out_channels=encoder_dim, kernel_size=3, stride=2)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(encoder_dim * 72, encoder_dim)
        self.dropout = nn.Dropout(p=0.5)

        #conformer block x N
        self.blocks = [ConformerBlock(dim_model=encoder_dim, attn_heads=att_heads, kernel_size=kernel_size).cuda() for x in range(num_blocks)]
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        t,b,h,w = out.shape
        print('the shape: ', out.reshape(t,w,b*h).shape)
        out = self.linear(out.reshape(t,w,b*h))
        print("shape before block: ", out.shape)
        for block in self.blocks:
            out = block(out)
        
        return out

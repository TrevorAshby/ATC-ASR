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
    def __init__(self, d_module, expansion_factor, kernel_size):
        self.layernorm = nn.LayerNorm(d_module)
        self.pointwise1 = nn.Conv1d(in_channels=d_module, 
                                   out_channels=d_module * expansion_factor, kernel_size=(1,1), padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(in_channels=d_module, out_channels=d_module, 
                                   kernel_size=kernel_size, stride=1, padding=((31-1)//2))
        self.batchnorm = nn.BatchNorm1d(num_features=d_module)
        # swish = x * x.sigmoid()
        self.pointwise2 = nn.Conv1d(in_channels=d_module, out_channels=d_module, 
                                    kernel_size=kernel_size, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.layernorm(x)
        out = self.pointwise1(out)
        out = self.glu(out)
        out = self.depthwise(out)
        out = self.batchnorm(out)
        out = out * out.sigmoid() # swish
        out = self.pointwise2(out)
        out = self.dropout(out)
        out += x
        return out
        
        
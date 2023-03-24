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

class ConvolutionModule(nn.Module):
    def __init__(self, d_module, expansion_factor, kernel_size):
        super(ConvolutionModule, self).__init__()
        self.layernorm = nn.LayerNorm(d_module)
        self.pointwise1 = nn.Conv1d(in_channels=d_module, 
                                   out_channels=d_module * expansion_factor, kernel_size=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(in_channels=d_module, out_channels=d_module, 
                                   kernel_size=kernel_size, padding=((kernel_size-1)//2))
        self.batchnorm = nn.BatchNorm1d(num_features=d_module)
        # swish = x * x.sigmoid()
        self.pointwise2 = nn.Conv1d(in_channels=d_module, out_channels=d_module, 
                                    kernel_size=1, padding=0, bias=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        out = self.layernorm(x)
        print("the out shape: ", out.shape)
        b,h,w = out.shape
        print('the shape 2: ', out.reshape(b,w,h).shape)
        out = self.pointwise1(out.reshape(b,w,h))
        out = self.glu(out)
        print("out before depthwise: ", out.shape)
        out = self.depthwise(out)
        out = self.batchnorm(out)
        out = out * out.sigmoid() # swish
        print("out before 2nd pointwise: ", out.shape)
        out = self.pointwise2(out)
        out = self.dropout(out)
        print("x shape: ", x.shape)
        print("out+x shape out: ", out.shape)
        out = out.reshape(b,h,w)
        out += x
        
        print("final shape: ", out.shape)
        return out
        
        
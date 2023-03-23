# layernorm
# linear layer
    # expansion factor of 4
# swish activation
# dropout
# linear layer
    # back to model dimension
# dropout
# + (x)

import torch
import torch.nn as nn

class FeedForwardModule(nn.Module):
    def __init__(self, d_module, expansion_factor, p_dropout=0.1):
        super(FeedForwardModule, self).__init__()
        self.layernorm = nn.LayerNorm(d_module)
        x1 = d_module * expansion_factor
        self.linear1 = nn.Linear(in_features=d_module, out_features=x1)
        self.dropout = nn.Dropout(p=p_dropout)
        x2 = expansion_factor * d_module
        self.linear2 = nn.Linear(in_features=x2, out_features=d_module)

    def forward(self, x):
        # x = (b, w, h)
        out = self.layernorm(x)
        out = self.linear1(out)
        out = out * out.sigmoid() # swish
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += x # residual with input
        return out # should be (b, w, h)
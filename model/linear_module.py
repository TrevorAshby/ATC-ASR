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

class FeedForwardModule():
    def __init__(self):
        self.layernorm = nn.LayerNorm(256)
        self.linear1 = nn.Linear(256, 1024)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024, 256)

    def forward(self, x):
        # x = (b, w, h)
        out = self.layernorm(x)
        out = self.linear1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        out += x # residual with input
        return out # should be (b, w, h)
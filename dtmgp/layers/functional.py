'''
wrapper for ReLU
'''

import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        kl = 0
        return F.relu(input[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class MinMax(nn.Module):
    def __init__(self, lengthscale=1., bias=False):
        """
        Implements Min-max normalization as an activation
        """
        super().__init__()
        self.lengthscale = lengthscale
        self.bias = bias

    def forward(self, x):
        """
        ------------------------
        Parameters:
        x: [n,d] size tensor, n is the number of the input, d is the dimension of the input

        ------------------------
        Returns:
        out: [n,d] size tensor, normalized by Min-Max scaler
        """
        if x.min() == x.max():
            return x
        else:
            out = self.lengthscale * (x - x.min()) / (x.max() - x.min())
            return out

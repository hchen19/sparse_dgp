'''
Tensor Markov Kernel activation function
'''


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class TMK(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self):
        pass
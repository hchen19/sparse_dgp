'''
Tensor Markov Kernel activation function
'''


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class TMK(nn.Module):
    def __init__(self,):
        """
        Implements tensor markov kernel as an activation 
        """
        super().__init__()
        
    
    def forward(self, input, kernel, sparse_grid, chol_inv):
        """
        ------------------------
        Parameters:
        input: [n,d] size tensor, n is the number of the input, d is the dimension of the input
        sparse_grid: [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        chol_inv: [m,m] size tensor, inverse of Cholesky decomposition of kernel(sparse_grid,sparse_grid),
                stored in torch.sparse_coo_tensor format

        ------------------------
        Returns:
        out: kernel(input, sparse_grid) @ chol_inv
        """
        
        k_star = kernel(input, sparse_grid) # [n,m] size tenosr
        out = k_star @ chol_inv # [n,m] size tensor

        return out
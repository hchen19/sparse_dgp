'''
Tensor Markov Kernel activation function
'''


import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


class TMK(nn.Module):
    def __init__(self, kernel, sparse_grid, chol_inv) -> None:
        """
        ------------------------
        Parameters:
        kernel: tensor markov kernel
        sparse_grid: sparse grid points X^{SG} of dyadic sort, is [m,d] size tensor
        chol_inv: inverse of Cholesky decomposition of kernel(sparse_grid,sparse_grid),
                stored in torch.sparse_coo_tensor format, [m,m] size tensor
        """
        super().__init__()
        self.kernel = kernel
        self.sparse_grid = sparse_grid
        self.chol_inv = chol_inv
        
    
    def forward(self, input):
        """
        ------------------------
        Parameters:
        input: [n,d] size tensor, n is the number of the input, d is the dimension of the input

        ------------------------
        Returns:
        phi: kernel(input, sparse_grid) @ chol_inv
        """
        kernel = self.kernel
        sparse_grid = self.sparse_grid
        chol_inv = self.chol_inv # [m,m] size tensor in torch.sparse_coo_tensor format

        k_star = kernel(input, sparse_grid) # [n,m] size tenosr
        phi = k_star @ chol_inv # [n,m] size tensor
        return phi
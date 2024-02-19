'''
Tensor Markov Kernel activation function
'''


import torch
import torch.nn as nn

from dtmgp.utils.sparse_activation.design_class import HyperbolicCrossDesign
from dtmgp.utils.sparse_activation.sparse_grid import SparseGridDesign
from dtmgp.kernels.laplace_kernel import LaplaceProductKernel
from dtmgp.utils.operators.chol_inv import mk_chol_inv, tmk_chol_inv


class TMK(nn.Module):
    def __init__(self, feature_dim=10, eta=3,
                 input_bd=None, design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        """
        Implements tensor markov kernel as an activation 
        """
        super().__init__()
        self.d = feature_dim
        self.eta = eta
        self.input_bd = input_bd
        self.design_class = design_class
        
        self.kernel = kernel
        self.sg = SparseGridDesign(feature_dim, eta, input_bd=input_bd, design_class=design_class).gen_sg(dyadic_sort=True, return_neighbors=True)
        self.pts_set = self.sg.pts_set

        self.chol_inv = tmk_chol_inv(sparse_grid_design=self.sg, 
                                     tensor_markov_kernel=self.kernel, 
                                     upper = True)
    
    def forward(self, input):
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
        
        k_star = self.kernel(input, self.pts_set) # [n,m] size tenosr
        out = k_star @ self.chol_inv # [n,m] size tensor

        return out
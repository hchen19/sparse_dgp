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
    def __init__(self, feature_dim=8, n_level=2,
                 input_bd=None, design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        """
        Implements tensor markov kernel as an activation 
        """
        super().__init__()

        self.kernel = kernel

        if feature_dim == 1: # one-dimension TMGP
            dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_bd=input_bd)
            chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel = kernel, upper= True)
            design_points = dyadic_design.points.reshape(-1,1)
        else: # multi-dimension TMGP
            eta = int(feature_dim + n_level)
            sg = SparseGridDesign(feature_dim, eta, input_bd=input_bd, design_class=design_class).gen_sg(dyadic_sort=True, return_neighbors=True)
            chol_inv = tmk_chol_inv(sparse_grid_design=sg, tensor_markov_kernel=kernel, upper = True)
            design_points = sg.pts_set
        
        # self.n_dim = feature_dim
        # self.n_level = n_level
        # self.input_bd = input_bd
        # self.design_class = design_class
        
        self.chol_inv = chol_inv # [n_points, n_points] size tensor
        self.design_points = design_points # [n_points, n_dim] size tensor
        self.out_features = design_points.shape[0] 
        
    
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
        
        k_star = self.kernel(input, self.design_points) # [n, m] size tenosr
        out = k_star @ self.chol_inv # [n, m] size tensor

        return out
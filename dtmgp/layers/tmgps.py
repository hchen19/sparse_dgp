"""
Tensor Markov Kernel activation function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtmgp.kernels.laplace_kernel import LaplaceProductKernel
from dtmgp.layers.functional import MinMax
from dtmgp.utils.sparse_activation.design_class import HyperbolicCrossDesign, SparseGridDesign
from dtmgp.utils.operators.chol_inv import mk_chol_inv, tmk_chol_inv


class tmgp_sg(nn.Module):
    def __init__(self, in_features, n_level=2,
                 input_bd=None, design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        """
        Implements tensor markov GP as an activation layer using sparse grid structure.

        Parameters:
            in_features: int -> size of each input sample,
            n_level: int -> level of sparse grid design,
            input_bd,
            design_class: class -> design class of sparse grid,
            kernel: class -> kernel function of deep GP
        """
        super().__init__()

        self.kernel = kernel
        self.scaler = MinMax()

        if in_features == 1:  # one-dimension TMGP
            dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_bd=input_bd)
            chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)
            design_points = dyadic_design.points.reshape(-1, 1)
        else:  # multi-dimension TMGP
            eta = int(in_features + n_level)
            sg = SparseGridDesign(in_features, eta, input_bd=input_bd, design_class=design_class).gen_sg(
                dyadic_sort=True, return_neighbors=True)
            chol_inv = tmk_chol_inv(sparse_grid_design=sg, tensor_markov_kernel=kernel, upper=True)
            design_points = sg.pts_set

        self.register_buffer('design_points', design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv', chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0]

    def forward(self, x):
        """
        ------------------------
        Parameters:
        x: [n,d] size tensor, n is the number of the input, d is the dimension of the input
        self.design_points: [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.chol_inv: [m,m] size tensor, inverse of Cholesky decomposition of kernel(sparse_grid,sparse_grid),
                stored in torch.sparse_coo_tensor format

        ------------------------
        Returns:
        out: [n,m] size tensor, kernel(input, sparse_grid) @ chol_inv
        """
        x = self.scaler(x)
        k_star = self.kernel(x, self.design_points)  # [n, m] size tenosr
        out = k_star @ self.chol_inv  # [n, m] size tensor

        return out


class tmgp_additive(nn.Module):
    def __init__(self, in_features, n_level,
                 input_bd=None, design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        """
        Implements additive markov GP as an activation layer using additive structure

        Parameters:
            in_features: int -> size of each input sample,
            input_bd,
            design_class: class -> design class of sparse grid,
            kernel: class -> kernel function of deep GP
        """
        super().__init__()

        self.in_features = in_features
        self.kernel = kernel
        self.scaler = MinMax()

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_bd=input_bd)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)
        design_points = dyadic_design.points.reshape(-1, 1)

        self.register_buffer('design_points', design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv', chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0] * in_features

    def forward(self, x):
        """
        Parameters:
        ------------------------
        x: [n,d=1] size tensor, n is the number of the input, d is the dimension of the input
        self.design_points: [m,d=1] size tensor, sparse grid points X^{SG} of dyadic sort
        self.chol_inv: [m,m] size tensor, inverse of Cholesky decomposition of kernel(sparse_grid,sparse_grid),
                stored in torch.sparse_coo_tensor format

        Returns:
        ------------------------
        out: [n,m] size tensor, kernel(input[:,dim], design_points) @ chol_inv
        """
        x = self.scaler(x)
        x = torch.permute(x, (1,0))
        x = torch.flatten(x)
        k_star = self.kernel(x, self.design_points)  # [...,n, m] size tenosr
        phi = k_star @ self.chol_inv  # [..., n, m] size tensor
        out = phi.reshape(-1, self.out_features)

        return out

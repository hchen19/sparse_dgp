import torch
import torch.nn as nn
import torch.nn.functional as F

from sparse_dgp.kernels.laplace_kernel import LaplaceProductKernel
from sparse_dgp.layers.functional import MinMax
from sparse_dgp.utils.sparse_activation.design_class import HyperbolicCrossDesign, SparseGridDesign
from sparse_dgp.utils.operators.chol_inv import mk_chol_inv, tmk_chol_inv

__all__ = [
    'TMGP',
    'AMGP',
]


class TMGP(nn.Module):
    """
    Implements tensor markov GP as an activation layer using sparse grid structure.

    .. math::

        \begin{equation*}
            k\left( \mathbf{x}, X^{SG} \right)R^{-1}
        \end{equation*}

    :param in_features: Size of each input sample.
    :type in_features: int
    :param n_level: Level of sparse grid design. (Default: `2`.)
    :type n_level: int, optional
    :param input_bd: Input boundary. (Default: `None`=[0,1].)
    :type input_bd: 2-size list
    :param design_class: Base design class of sparse grid. (Default: `HyperbolicCrossDesign`.)
    :type design_class: class, optional
    :param kernel: Kernel function of deep GP. (Default: `LaplaceProductKernel(lengthscale=1.)`.)
    :type kernel: class, optional
    """

    def __init__(self,
                 in_features,
                 n_level=2,
                 input_lb=-2,
                 input_ub=2,
                 design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        super().__init__()

        self.kernel = kernel
        # self.ln1 = MinMax()
        self.ln1 = nn.LayerNorm(in_features)

        if in_features == 1:  # one-dimension TMGP
            dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb, input_ub=input_ub)
            chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)
            design_points = dyadic_design.points.reshape(-1, 1)
        else:  # multi-dimension TMGP
            eta = int(in_features + n_level)
            sg = SparseGridDesign(in_features, eta, input_lb=input_lb, input_ub=input_ub, design_class=design_class).gen_sg(
                dyadic_sort=True, return_neighbors=True)
            chol_inv = tmk_chol_inv(sparse_grid_design=sg, tensor_markov_kernel=kernel, upper=True)
            design_points = sg.pts_set

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0]

    def forward(self, x):
        """
        Computes the tensor markov kernel of :math:`\mathbf x`.

        :param x: [n,d] size tensor, n is the number of the input, d is the dimension of the input

        :return: [n,m] size tensor, kernel(input, sparse_grid) @ chol_inv
        """
        out = self.ln1(x)
        out = self.kernel(out, self.design_points)  # [n, m] size tenosr
        out = out @ self.chol_inv  # [n, m] size tensor

        return out


class AMGP(nn.Module):
    """
    Implements tensor markov GP as an activation layer using additive structure.

    .. math::

        \begin{equation*}
            \left\{ k\left( x_i, X^{SG} \right)R^{-1} \right\}^{d}_{i=1}
        \end{equation*}

    :param in_features: Size of each input sample.
    :type in_features: int
    :param n_level: Level of sparse grid design. (Default: `2`.)
    :type n_level: int, optional
    :param input_bd: Input boundary. (Default: `None`=[0,1].)
    :type input_bd: 2-size list
    :param design_class: Base design class of sparse grid. (Default: `HyperbolicCrossDesign`.)
    :type design_class: class, optional
    :param kernel: Kernel function of deep GP. (Default: `LaplaceProductKernel(lengthscale=1.)`.)
    :type kernel: class, optional
    """

    def __init__(self,
                 in_features,
                 n_level,
                 input_lb=-2,
                 input_ub=2,
                 design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        super().__init__()

        self.kernel = kernel
        # self.ln1 = MinMax()
        self.ln1 = nn.LayerNorm(in_features)

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb, input_ub=input_ub)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)  # [m, m] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [m, 1] size tensor

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0] * in_features  # m*d

    def forward(self, x):
        """
        Computes the element-wise tensor markov kernel of :math:`\mathbf x`.

        :param x: [n,d] size tensor, n is the number of the input, d is the dimension of the input

        :return: [n,m*d] size tensor, kernel(input, sparse_grid) @ chol_inv
        """

        out = self.ln1(x)
        out = torch.flatten(out, start_dim=1)  # flatten x of size [...,n,d] --> size [...,n*d]
        out = out.unsqueeze(dim=-1)  # add new dimension, x of size [...,n*d] --> size [...,n*d, 1]
        out = self.kernel(out, self.design_points)  # [...,n*d, m] size tenosr
        out = torch.matmul(out, self.chol_inv)  # [..., n*d, m] size tensor
        out = torch.flatten(out, start_dim=1)

        return out

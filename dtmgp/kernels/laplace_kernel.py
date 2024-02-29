from typing import Optional

import torch
from torch import Tensor


class LaplaceProductKernel(torch.nn.Module):
    """
    ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if :math:`\mathbf x` is a `... x N x D` matrix. (Default: `None`.)
    
    active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    """
    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

        self.x1 = None
        self.x2 = None
        self.d = None
        self.distance = None

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        """
        Computes the covaraince between x1 and x2.
        ------------------------
        Parameters:
        x1: [n, d] size tensor
        x2: [m, d] size tensor
        diag: Should the kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)

        ------------------------
        Returns: the kernel matrix or vector. The shape depends on the kernel's evaluation mode:
            * 'full_covar`: `n x m`
            * `diag`: `n`
        """

        # Give x1 and x2 a last dimension, if necessary
        if x1.ndimension() == 1:
            self.x1 = x1.unsqueeze(1)
        if x2 is not None:
            if x2.ndimension() == 1:
                self.x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        
        if x2 is None:
            self.x2 = self.x1
        

        # lengthscale
        d = self.x1.shape[-1]
        if self.lengthscale is None:
            self.lengthscale = torch.ones(d)*d
        if isinstance(self.lengthscale, (int, float)):
            self.lengthscale *= torch.ones(d)
        if isinstance(self.lengthscale, Tensor):
            if self.lengthscale.ndimension() == 0 or max(self.lengthscale.size()) == 1:
                self.lengthscale *= torch.ones(d)
            if not max(self.lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        self.lengthscale = self.lengthscale.reshape(-1)

        
        adjustment = self.x1.mean(dim=-2, keepdim=True) # [d] size tensor
        x1_ = (self.x1 - adjustment).div(self.lengthscale)
        x2_ = (self.x2 - adjustment).div(self.lengthscale)
        #distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                self.distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype)
            else:
                self.distance = torch.sum(torch.abs(x1_-x2_),dim=-1)
        else:
            self.distance = torch.cdist(x1_, x2_, p=1)
            self.distance = self.distance.clamp_min(1e-15)

        res = torch.exp( -self.distance )
        return res
    

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype)
            else:
                res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
                return res.pow(2) if square_dist else res
        else:
            dist_func = sq_dist if square_dist else dist
            return dist_func(x1, x2, x1_eq_x2)


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-2, dim2=-1).fill_(0)

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    if not x1_eq_x2:
        res = torch.cdist(x1, x2)
        return res.clamp_min(1e-15)
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()
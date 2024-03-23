from typing import Optional

import torch
from torch import Tensor


class LaplaceProductKernel(torch.nn.Module):
    def __init__(self, lengthscale=None):
        """
        Implements Laplace Product Kernel.

        Parameters:
            lengthscale: None -> [d] size tensor
        """
        super().__init__()
        self.lengthscale = lengthscale

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
        lengthscale = self.lengthscale

        # Give x1 and x2 a last dimension, if necessary
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # lengthscale
        d = x1.shape[-1]

        if lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)

        if isinstance(lengthscale, (int, float)):
            # if lengthscale is int or float, such as lengthscale = 3 or 3.0
            # let lenthscale be d-dimensional torch.Tensor with same value, such as torch.tensor([3.0, 3.0,.., 3.0])
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)
        
        if isinstance(lengthscale, Tensor):
            # if lengthscale is a 0-size or one-dim torch.Tensor, such as lengthscale = torch.tensor(3), torch.tensor(3.) or torch.tensor([3]), torch.tensor([3.0])
            # let lengthscale be d-dimensional torch.Tensor with same value, such as lengthscale = torch.tensor([3.0, 3.0,.., 3.0])
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            
            # if dimension of lengthscale is not d, such as lengthscale = torch.tensor([3., 3., 3., 3.]) but d is not equal to 4
            # raise Error
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        #distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.sum(torch.abs(x1_-x2_), dim=-1)
        else:
            distance = torch.cdist(x1_, x2_, p=1)
            distance = distance.clamp_min(1e-15)

        res = torch.exp(-distance)
        return res


class LaplaceAdditiveKernel(torch.nn.Module):
    """
    lengthscale: [d] size tensor
    """
    def __init__(self, lengthscale=None):
        """
        Implements Laplace Additive Kernel.

        Parameters:
            lengthscale: None -> [d] size tensor
        """
        super().__init__()
        self.lengthscale = lengthscale

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
        lengthscale = self.lengthscale

        # Give x1 and x2 a last dimension, if necessary
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # lengthscale
        d = x1.shape[-1]
        if lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)

        if isinstance(lengthscale, (int, float)):
            # if lengthscale is int or float, such as lengthscale = 3 or 3.0
            # let lenthscale be d-dimensional torch.Tensor with same value, such as torch.tensor([3.0, 3.0,.., 3.0])
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)
        
        if isinstance(lengthscale, Tensor):
            # if lengthscale is a 0-size or one-dim torch.Tensor, such as lengthscale = torch.tensor(3), torch.tensor(3.) or torch.tensor([3]), torch.tensor([3.0])
            # let lengthscale be d-dimensional torch.Tensor with same value, such as lengthscale = torch.tensor([3.0, 3.0,.., 3.0])
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            
            # if dimension of lengthscale is not d, such as lengthscale = torch.tensor([3., 3., 3., 3.]) but d is not equal to 4
            # raise Error
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True) # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        #distance = self.covar_dist(x1_, x2_, diag=diag, **params)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.abs(x1_-x2_)
        else:
            distance = x1_.unsqueeze(dim=-2).repeat(*x1_.shape[:-2],1,x2_.shape[-2],1) - x2_.unsqueeze(dim=-3).repeat(*x2_.shape[:-2],x1_.shape[-2],1,1)
            #distance = distance.clamp_min(1e-15)

        res = torch.sum(torch.exp(-distance), dim=-1)
        return res

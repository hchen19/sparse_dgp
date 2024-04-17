'''
wrapper for ReLU
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ReLU',
    'MinMax',
]


class ReLU(nn.Module):
    """
    Implement ReLU activation used in BNN. 

    .. math::

        \begin{equation*}
            \text{ReLU}(x)=(x)^{+}=\max\left( 0,x \right)
        \end{equation*}

    Args:
        inplace (bool, optional):
            can optionally do the operation in-place. It should be a [d] size tensor. Default: `False`.
    """

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


class ReLUN(nn.Module):
    """
    Implement ReLU-N activation used in BNN.

    .. math::

        \begin{equation*}
            \text{ReLU-N}(x)=\min\left( \max\left( 0,x \right),N \right)
        \end{equation*}

    Args:
        upper (float, optional):
            Set this if you want a customized upper bound of ReLU. Default: `1.0`.
        inplace (bool, optional):
            can optionally do the operation in-place. It should be a [d] size tensor. Default: `False`.
    """

    def __init__(self, upper=1, inplace=False):
        super(ReLUN, self).__init__()
        self.inplace = inplace
        self.upper = upper

    def forward(self, input):
        ub = torch.ones_like(input, dtype=input.dtype, device=input.device) * self.upper
        input = torch.min(input, ub)
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class MinMax(nn.Module):
    """
    Implement Min-Max normalization. 

    .. math::

        \begin{equation*}
            x_{\text{scaled}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}} \ast \gamma + \beta
        \end{equation*}
    
    where :math:`\gamma` is the lengthscale parameter, :math:`\beta` is the bias parameter.

    Args:
        lengthscale (float, optional):
            Set this if you want a customized lengthscale. Default: `1.0`.
        bias (float, optional):
            Set this if you want a customized bias. Default: `None`.
    """

    def __init__(self, lengthscale=1., bias=0., eps=1e-05):
        super().__init__()
        self.lengthscale = lengthscale
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        """
        Computes the Min-Max normalization of :math:`\mathbf x`.

        :param x1: data to be normalized ([n, d] size tensor).
        
        :return: normalized data :math:`\mathbf x`.
        """
        if (x.max() - x.min()) > self.eps:
            out = self.lengthscale * (x - x.min()) / (x.max() - x.min()) + self.bias
        else:
            out = x / x.max() + self.bias
        return out


def minmax(input, lengthscale=1., bias=0., eps=1e-05):
    return MinMax(lengthscale, bias, eps)(input)

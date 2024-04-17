'''
wrapper for ReLU
'''

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
        if (x.max() - x.min()) < self.eps:
            out = x / x.max() + self.bias
        else:
            out = self.lengthscale * (x - x.min()) / (x.max() - x.min()) + self.bias
        return out


def relu(input, lengthscale=1., bias=0., eps=1e-05):
    return MinMax(lengthscale, bias, eps)(input)

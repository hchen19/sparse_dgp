import torch

def hc_design(deg, input_bd):
    r"""
    ------------------------
    Parameters:
    deg: degree of hyperbolic cross (# of points = 2^j - 1)
    input_bd: [x_leftbd, x_rightbd], [2] size tensor

    ------------------------
    Returns:
    res: [1/(2^deg), 2/(2^deg), 3/(2^deg),..., (2^deg-1)/(2^deg)]
         [2^deg-1] size tensor with hyperbolic cross points (bisection)
         
    """
    x_1 = input_bd[0]
    x_n = input_bd[1]
    res = ( torch.arange(start=2**deg-1, end=0, step=-1) * x_1 + \
           torch.arange(start=1, end=2**deg, step=1) * x_n ) * (0.5**deg)
    
    return res
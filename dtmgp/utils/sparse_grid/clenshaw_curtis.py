import torch

def cc_design(deg, input_bd):
    r"""
    ------------------------
    Parameters:
    deg: degree of clenshaw curtis (# of points = 2^(deg-1) + 1)
    input_bd: [x_leftbd, x_rightbd], [2] size tensor

    ------------------------
    Returns:
    m_i = 2^(i-1) + 1
    res: [-cos( 0*pi/ m_i-1 ), -cos( 1*pi/ m_i-1 ), ..., -cos( (m_i-1)*pi/ m_i-1 ) ]
         
    """
    x_1 = input_bd[0]
    x_n = input_bd[1]
    m_i = 2**(deg) + 1
    res_basis = - torch.cos( torch.pi * torch.arange(1, m_i-1) / (m_i-1)) # interval on [-1,1]
    res = res_basis*(x_n-x_1)/2 + (x_n+x_1)/2 # [m_i] siz tensor
    return res
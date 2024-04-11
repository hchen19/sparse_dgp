from .mnist_dgp_add_variational import DAMGPmnist
from .mnist_dgp_sg_variational import DTMGPmnist
from .simple_cnn_variational import SCNN
from .simple_dgp_add_variational import SDTMGPadd
from .simple_dgp_sg_variational import SDTMGPsg
from .simple_fc_variational import SFC

__all__ = [
    "DAMGPmnist",
    "DTMGPmnist",
    "SCNN",
    "SDTMGPadd",
    "SDTMGPsg",
    "SFC",
]
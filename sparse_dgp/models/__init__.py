from .mnist_dgp_add_variational import DAMGPmnist
from .mnist_dgp_sg_variational import DTMGPmnist
from .mnist_cnn_variational import SCNN
from .simple_dgp_add_variational import SDTMGPadd
from .simple_dgp_sg_variational import SDTMGPsg
from .simple_fc_variational import SFC
from .cifar_resnet_variational import *
from .cifar_dgp_variational import *
from .cifar_resgp_variational import *
from .imgnet_resnet_variational import *

__all__ = [
    "DAMGPmnist",
    "DTMGPmnist",
    "SCNN",
    "SDTMGPadd",
    "SDTMGPsg",
    "SFC",
]
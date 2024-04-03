from .mnist_dtmgp_add import DtmgpAdditive
from .mnist_dtmgp_sg import DtmgpSparsegrid
from .simple_cnn_variational import SCNN
from .simple_dtmgp_add_variational import SDTMGPadd
from .simple_dtmgp_sg_variational import SDTMGPsg
from .simple_fc_variational import SFC

__all__ = [
    "DtmgpAdditive",
    "DtmgpSparsegrid",
    "SCNN",
    "SDTMGPadd",
    "SDTMGPsg",
    "SFC",
]
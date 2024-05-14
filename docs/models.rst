.. role:: hidden
    :class: hidden-section

sparse_dgp.models
===================================

.. automodule:: sparse_dgp.models
.. currentmodule:: sparse_dgp.models


Models for MNIST
-----------------------------

:hidden:`DAMGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: DAMGPmnist # dtmgp/models/mnist_dgp_add_variational.py
   :members:


:hidden:`DTMGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: DTMGPmnist # dtmgp/models/mnist_dgp_sg_variational.py
   :members:

Models for Simple Dataset
-----------------------------------

:hidden:`DAMGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SDTMGPadd # dtmgp/models/simple_dgp_add_variational.py
   :members:


:hidden:`DTMGP`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SDTMGPsg # dtmgp/models/simple_dgp_sg_variational.py
   :members:

:hidden:`CNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SCNN # dtmgp/models/simple_cnn_variational.py
   :members:

:hidden:`FullyconnectedNN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SFC # dtmgp/models/simple_fc_variational.py
   :members:
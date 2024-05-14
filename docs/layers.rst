.. role:: hidden
    :class: hidden-section

sparse_dgp.layers
===================================

.. automodule:: sparse_dgp.layers
.. currentmodule:: sparse_dgp.layers


One-layer Markov GP
-----------------------------

:hidden:`AdditiveMarkovGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: AMGP # dtmgp/layers/tmgp.py
   :members:


:hidden:`TensorMarkovGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: TMGP # dtmgp/layers/tmgp.py
   :members:


Base Variational Layer
-----------------------------------

:hidden:`_BaseVariationalLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: _BaseVariationalLayer # dtmgp/layers/base_variational_layer.py
   :members:


Linear Layer
-----------------------------------

:hidden:`LinearReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: LinearReparameterization # dtmgp/layers/linear.py
   :members:


Convolutional Layer
-----------------------------------

:hidden:`Conv1dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Conv1dReparameterization # dtmgp/layers/conv.py
   :members:

:hidden:`Conv2dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Conv2dReparameterization # dtmgp/layers/conv.py
   :members:

:hidden:`Conv3dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Conv3dReparameterization# dtmgp/layers/conv.py
   :members:

:hidden:`ConvTranspose1dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvTranspose1dReparameterization # dtmgp/layers/conv.py
   :members:

:hidden:`ConvTranspose2dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvTranspose2dReparameterization # dtmgp/layers/conv.py
   :members:

:hidden:`ConvTranspose3dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConvTranspose3dReparameterization # dtmgp/layers/conv.py
   :members:


Activation Function
-----------------------------------

:hidden:`ReLU`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReLU # dtmgp/layers/functional.py
   :members:

:hidden:`ReLUN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ReLUN # dtmgp/layers/functional.py
   :members:


Normalization Function
-----------------------------------

:hidden:`MinMax`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: MinMax # dtmgp/layers/functional.py
   :members:
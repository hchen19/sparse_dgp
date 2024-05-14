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

.. autoclass:: tmgp.AMGP # dtmgp/layers/tmgp.py
   :members:
   :private-members:
   :special-members:


:hidden:`TensorMarkovGP`
~~~~~~~~~~~~~~~~~

.. autoclass:: tmgp.TMGP # dtmgp/layers/tmgp.py
   :members:
   :private-members:
   :special-members:



Base Variational Layer
-----------------------------------

:hidden:`_BaseVariationalLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: base_variational_layer._BaseVariationalLayer # dtmgp/layers/base_variational_layer.py
   :members:


Linear Layer
-----------------------------------

:hidden:`LinearReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: linear.LinearReparameterization # dtmgp/layers/linear.py
   :members:
   :private-members:
   :special-members:



Convolutional Layer
-----------------------------------

:hidden:`Conv1dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.Conv1dReparameterization # dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:


:hidden:`Conv2dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.Conv2dReparameterization # dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:


:hidden:`Conv3dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.Conv3dReparameterization# dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:


:hidden:`ConvTranspose1dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.ConvTranspose1dReparameterization # dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:


:hidden:`ConvTranspose2dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.ConvTranspose2dReparameterization # dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:


:hidden:`ConvTranspose3dReparameterization`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: conv.ConvTranspose3dReparameterization # dtmgp/layers/conv.py
   :members:
   :private-members:
   :special-members:



Activation Function
-----------------------------------

:hidden:`ReLU`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: functional.ReLU # dtmgp/layers/functional.py
   :members:
   :private-members:
   :special-members:


:hidden:`ReLUN`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: functional.ReLUN # dtmgp/layers/functional.py
   :members:
   :private-members:
   :special-members:



Normalization Function
-----------------------------------

:hidden:`MinMax`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: functional.MinMax # dtmgp/layers/functional.py
   :members:
   :private-members:
   :special-members:

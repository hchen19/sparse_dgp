Basic Usage
=====

.. _installation:

Installation
------------

To use SparseDeepGP, first install it using pip:

Install from `GitHub <https://github.com/hchen19/sparse_dgp>`_
~~~~~~~~~~~~~~~~~

.. code-block:: console

   $ git clone https://github.com/hchen19/sparse_dgp.git
   $ cd sparse_dgp
   $ pip install -e .
   # pip install -r requirements.txt # install requirements

Install from `Package <https://test.pypi.org/project/sparse-dgp/>`_
~~~~~~~~~~~~~~~~~

.. code-block:: console

   (.venv) $ pip install sparse-dgp

Defining an example model
------------

In the next cell, we define a simple 2-layer sparse DGP for a regression task. We'll be using this model to demonstrate
the usage of the library.

.. code-block:: python

    import torch
    import sparse_dgp as gp

:github_url: https://github.com/hchen19/sparse_dgp

SparseDeepGP's documentation
====================================

**SparseDeepGP** is a Python library for sparse deep Gaussian processes with GPU acceleration. It is built on top of
`PyTorch <https://pytorch.org/>`_ and provides a simple and flexible API for building complex deep GP models.
This documentation is for the `GitHub Repo`_.

.. _GitHub Repo: https://github.com/hchen19/sparse_dgp


.. _installation:

Installation
------------

To use SparseDeepGP, make sure you have `PyTorch installed <https://pytorch.org/get-started/locally/>`_, then install it using pip:

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

.. note::

   This project is under active development.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Tutorials:

   intro.rst
   install.rst

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples:

   examples/**/index

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API

   api.rst

References
======================

* Liang Ding, Rui Tuo, and Shahin Shahrampour. `A Sparse Expansion For Deep Gaussian Processes`_. IISE Transactions (2023): 1-14. `Code <https://github.com/ldingaa/DGP_Sparse_Expansion>`_ in MATLAB version.
* Rishabh Agarwal, et al. `Neural Additive Models: Interpretable Machine Learning with Neural Nets <https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf>`_. Advances in neural information processing systems 34 (2021): 4699-4711.
* Wei Zhang, Brian Barr, and John Paisley. `Gaussian Process Neural Additive Models`_. AAAI Conference on Artificial Intelligence (2024)

.. _A Sparse Expansion For Deep Gaussian Processes: https://www.tandfonline.com/doi/pdf/10.1080/24725854.2023.2210629
.. _Gaussian Process Neural Additive Models: https://arxiv.org/pdf/2402.12518.pdf
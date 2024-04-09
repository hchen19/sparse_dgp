# Sparse Deep GPs
This repository is a Python implementation of sparse deep GP algorithms in PyTorch.

## Datasets
[MNIST](https://pytorch.org/vision/0.17/generated/torchvision.datasets.MNIST.html)

## Models

### Deep GPs with sparse grid


### Deep GPs with additive model

## Usage

## Examples
### MNIST
```bash
$ cd examples/mnist
$ python bayesian_dtmgp_mnist.py --model [additive_grid_model]
                                --barplot [plot_errorbar]
                                --batch-size [batch_size]
                                --test-batch-size [test_batch_size]
                                --epochs [epochs]
                                --lr [learning_rate]
                                --gamma [learning_rate_step_gamma]
                                --no-cuda [disable_cuda]
                                --seed [random_seed]
                                --log-interval [num_batches_log]
                                --save_dir [save_directory]
                                --mode [train_test_mode]
                                --num_monte_carlo [num_monte_carlo_inference]
                                --num_mc [num_monte_carlo_training]
                                --tensorboard [tensorboard_action]
                                --log_dir [logs_directory]
```

### Arguments
- `--model`: 'additive' or 'grid' (default: 'grid')
- `--barplot`: plot the errorbar of some examples, run in `--mode test` mode (default: True)
- `--batch-size`: Input batch size for training (default: 64)
- `--test-batch-size`: Input batch size for testing (default: 10000)
- `--epochs`: 'Number of epochs to train (default: 14)
- `--lr`: Learning rate (default: 1.0)
- `--gamma`: Learning rate step gamma (default: 0.7)
- `--no-cuda`: Disables CUDA training (default: False)
- `--seed`: Random seed (default: 10)
- `--log-interval`: Number of batches to wait before logging training status (default: 10)
- `--save_dir`: save directory (default: "examples/checkpoint/bayesian")
- `--mode`: mode = 'train' or 'test' (default: 'train')
- `--num_monte_carlo`: Number of Monte Carlo samples to be drawn for inference (default: 20)
- `--num_mc`: Number of Monte Carlo runs during training (default: 1)
- `--tensorboard`: Use tensorboard for logging and visualization of training progress
- `--log_dir`: Use tensorboard for logging and visualization of training progress (default: "examples/logs/main_bnn")


**Note**: to plot the barplots of some examples, the larger values of `--num_monte_carlo` is recommended, i.e.
```bash
$ cd examples/mnist
$ python bayesian_dtmgp_mnist.py --model additive --mode train
$ python bayesian_dtmgp_mnist.py --model additive --mode test --num_monte_carlo 100 --barplot True
```

## References
### GP
1. Liang Ding, Rui Tuo, and Shahin Shahrampour. [A Sparse Expansion For Deep Gaussian Processes](https://www.tandfonline.com/doi/pdf/10.1080/24725854.2023.2210629). IISE Transactions (2023): 1-14. [Code](https://github.com/ldingaa/DGP_Sparse_Expansion) in MATLAB version.
2. Rishabh Agarwal, et al. [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf). Advances in neural information processing systems 34 (2021): 4699-4711.
3. Wei Zhang, Brian Barr, and John Paisley. [Gaussian Process Neural Additive Models](https://arxiv.org/pdf/2402.12518.pdf). AAAI Conference on Artificial Intelligence (2024)

### UQ
1. Charlie Hewitt. [Confidence measures for CNN classification using Gaussian processes](https://chewitt.me/Papers/CTH-CNN-Conf-2018.pdf). (2018)
2. Yaniv Romano, Matteo Sesia, and Emmanuel Candes. [Classification with Valid and Adaptive Coverage](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf). Advances in Neural Information Processing Systems 33 (2020): 3581-3591.
3. Anastasios Angelopoulos, Stephen Bates, Jitendra Malik, and Michael I. Jordan. [Uncertainty Sets for Image Classifiers using Conformal Prediction](https://openreview.net/pdf?id=eNdiU_DbM9). ICLR (2021). [[Blog]](https://people.eecs.berkeley.edu/~angelopoulos/blog/posts/conformal-classification/)  [[Code]](https://github.com/aangelopoulos/conformal_classification)
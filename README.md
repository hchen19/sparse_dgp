# Sparse Deep GPs
This repository is a Python implementation of sparse deep GP algorithms in PyTorch.
## Datasets

## Models
### Deep GPs with sparse grid

### Deep GPs with additive model

## Examples
```bash
$ cd examples/test_data
$ python bayesian_dtmgp_additive.py --inputdim [input_dim] 
                                    --batch-size [batch_size]
                                    --test-batch-size [test_batch_size]
                                    --epochs [epochs]
                                    --lr [learning_rate]
                                    --gamma [learning_rate_step_gamma]
                                    --no-cuda [disable_cuda]
                                    --seed [random_seed]
                                    --save_dir [save_directory]
                                    --mode [train_test_mode]
                                    --num_monte_carlo [num_monte_carlo_inference]
                                    --num_mc [num_monte_carlo_training]
                                    --tensorboard [tensorboard_action]
                                    --log_dir [logs_directory]
```

### Arguments
- `--inputdim`: Input dim size for training (default: 7)
- `--batch-size`: Input batch size for training (default: 64)
- `--test-batch-size`: Input batch size for testing (default: 1000)
- `--epochs`: 'Number of epochs to train (default: 100)
- `--lr`: Learning rate (default: 1.0)
- `--gamma`: Learning rate step gamma (default: 0.999)
- `--no-cuda`: Disables CUDA training (default: False)
- `--seed`: Random seed (default: 1)
- `--save_dir`: save directory (default: "examples/checkpoint/bayesian")
- `--mode`: mode = 'train' or 'test' (default: 'train')
- `--num_monte_carlo`: Number of Monte Carlo samples to be drawn for inference (default: 20)
- `--num_mc`: Number of Monte Carlo runs during training (default: 5)
- `--tensorboard`: Use tensorboard for logging and visualization of training progress
- `--log_dir`: Use tensorboard for logging and visualization of training progress (default: "examples/logs/main_bnn")

## References
1. Ding, Liang, Rui Tuo, and Shahin Shahrampour. [A Sparse Expansion For Deep Gaussian Processes](https://www.tandfonline.com/doi/pdf/10.1080/24725854.2023.2210629). IISE Transactions (2023): 1-14. [Code](https://github.com/ldingaa/DGP_Sparse_Expansion) in MATLAB version.
2. Agarwal, Rishabh, et al. [Neural additive models: Interpretable machine learning with neural nets](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf). Advances in neural information processing systems 34 (2021): 4699-4711.



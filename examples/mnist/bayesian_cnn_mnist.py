from __future__ import print_function
import os
import sys
from pathlib import Path  # if you haven't already done so

file = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
import sparse_dgp.models.mnist_cnn_variational as simple_cnn

len_trainset = 60000
len_testset = 10000


def train(args, model, device, train_loader, optimizer, epoch, tb_writer=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output_ = []
        kl_ = []
        for mc_run in range(args.num_mc):
            output, kl = model(data)
            output_.append(output)
            kl_.append(kl)
        output = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)
        nll_loss = F.nll_loss(output, target)
        #ELBO loss
        loss = nll_loss + (kl / args.batch_size)

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss', loss.item(), epoch)
            tb_writer.flush()


def test(args, model, device, test_loader, epoch, tb_writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, kl = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() + (
                kl / args.batch_size)  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    val_accuracy = correct / len(test_loader.dataset)
    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', test_loss, epoch)
        tb_writer.add_scalar('val/accuracy', val_accuracy, epoch)
        tb_writer.flush()


def evaluate(args, model, device, test_loader):
    pred_probs_mc = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            for mc_run in range(args.num_monte_carlo):
                model.eval()
                output, _ = model.forward(data)
                #get probabilities from log-prob
                pred_probs = torch.exp(output)
                pred_probs_mc.append(pred_probs.cpu().data.numpy())

        target_labels = target.cpu().data.numpy()
        pred_mean = np.mean(pred_probs_mc, axis=0)
        pred_std = np.std(pred_probs_mc, axis=0)
        Y_pred = np.argmax(pred_mean, axis=1)
        print('Test accuracy:', (Y_pred == target_labels).mean() * 100)
        np.save('./probs_mnist_mc.npy', pred_probs_mc)
        np.save('./mnist_test_labels_mc.npy', target_labels)

        # plot some randomly selected examples
        # To plot the errorbar, set "--num_monte_carlo 100" or the higher values in argument would be recommended
        barplot = True
        if barplot:
            num_examples = 10
            # randomly select [num_samples]-size indices from torch.arange(0,len_testset) without replacement
            indices = torch.randperm(data.shape[0])[:num_examples]

            # data is [len_testset, 1, 28, 28] size tensor
            # target is [len_testset] size tensor
            data_examples = data[indices, :, :, :].cpu()  # [num_examples, 1, 28, 28] size tensor
            target_examples = target[indices].cpu()  # [num_examples] size tensor

            pred_mean_examples = pred_mean[indices, :]  # [num_examples, 10] size tensor
            pred_std_examples = pred_std[indices, :]  # [num_examples, 10] size tensor

            for i in range(num_examples):
                fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

                # plot true digits
                axes[0].imshow(data_examples[i, 0, :, :], cmap='gray', interpolation='nearest')

                # plot bar chart with std over digits 0,...,9
                axes[1].bar(np.arange(10), pred_mean_examples[i, :], color='lightblue')
                axes[1].errorbar(np.arange(10), pred_mean_examples[i, :], yerr=pred_std_examples[i, :],
                                 fmt='.', color='red', elinewidth=2, capthick=10, errorevery=1,
                                 alpha=0.5, ms=4, capsize=2)
                axes[1].set_xlabel('digits', fontsize=20)
                axes[1].set_ylabel('digits prob', fontsize=20)
                axes[1].tick_params(labelsize=20)

                # major ticks every 1, minor ticks every 5
                xmajor_ticks = np.arange(0, 10, 1)
                ymajor_ticks = np.arange(0, 1.1, 0.1)

                axes[1].set_xticks(xmajor_ticks)
                axes[1].set_yticks(ymajor_ticks)
                axes[1].grid(which='both', linestyle='--', linewidth=1.5)

                plt.close(fig)
                # save the full figure
                fig.savefig(f'./figures/barplots/cnn_barplot_{i}.png')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--error-bar',
                        type=bool,
                        default=True,
                        help='plot the error bar')
    parser.add_argument('--subset-size',
                        type=int,
                        default=60000,
                        metavar='N',
                        help='the size of the training subset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=10000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./checkpoint/bayesian')
    parser.add_argument('--mode', type=str, default='train', help='train | test')
    parser.add_argument(
        '--num_monte_carlo',
        type=int,
        default=20,
        metavar='N',
        help='number of Monte Carlo samples to be drawn for inference')
    parser.add_argument('--num_mc',
                        type=int,
                        default=1,
                        metavar='N',
                        help='number of Monte Carlo runs during training')
    parser.add_argument(
        '--tensorboard',
        action="store_true",
        help=
        'use tensorboard for logging and visualization of training progress')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/mnist/bayesian',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    tb_writer = None
    if args.tensorboard:

        logger_dir = os.path.join(args.log_dir, 'tb_logger')
        print("yee")
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        tb_writer = SummaryWriter(logger_dir)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])
    mnist_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    # Create a subset of the MNIST dataset
    if args.subset_size < len_trainset:
        subset_size = args.subset_size
        subset_indices = torch.randperm(len(mnist_dataset))[:subset_size]
        subset_mnist = Subset(mnist_dataset, subset_indices)

        # Create the subset DataLoader
        train_loader = DataLoader(subset_mnist, batch_size=args.batch_size, shuffle=True, **kwargs)

    else:
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST(
        '../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])),
                                              batch_size=args.test_batch_size,
                                              shuffle=False,
                                              **kwargs)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = simple_cnn.SCNN()
    model = model.to(device)

    start = time.time()
    print(args.mode)
    if args.mode == 'train':

        for epoch in range(1, args.epochs + 1):
            if (epoch < args.epochs / 2):
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
            else:
                optimizer = optim.Adadelta(model.parameters(), lr=args.lr / 10)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
            train(args, model, device, train_loader, optimizer, epoch,
                  tb_writer)
            test(args, model, device, test_loader, epoch, tb_writer)
            scheduler.step()

            torch.save(model.state_dict(),
                       args.save_dir + "/mnist_bayesian_scnn.pth")

    elif args.mode == 'test':
        checkpoint = args.save_dir + '/mnist_bayesian_scnn.pth'
        model.load_state_dict(torch.load(checkpoint))
        evaluate(args, model, device, test_loader)

    end = time.time()
    print("done. Total time: " + str(end - start))


if __name__ == '__main__':
    main()
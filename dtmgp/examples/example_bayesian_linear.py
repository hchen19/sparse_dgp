from __future__ import print_function
import os
import argparse
import sys
from pathlib import Path # if you haven't already done so
file = Path(os.path.dirname(os.path.abspath("__file__"))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import dtmgp.models.simple_fc_variational as simple_bnn
from dataset.dataset import Dataset
import matplotlib.pyplot as plt

class BNN:
    def __init__(self, input_dim, output_dim=1,
                 num_mc=1, num_monte_carlo=10, batch_size=128,
                 lr=1.0, 
                 gamma=0.999, 
                 activation=F.relu, 
                 inverse_y=False, 
                 seed=1,
                 use_cuda=True,
                 ):

        if torch.cuda.is_available() and use_cuda:
            self.device = torch.device('cuda:0')
            print("Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        torch.manual_seed(seed)

        self.lr = lr
        self.gamma = gamma
        self.inverse_y = inverse_y

        self.batch_size = batch_size
        self.num_monte_carlo = num_monte_carlo
        self.num_mc = num_mc

        self.activation = activation

        self.model = simple_bnn.SFC(input_dim, output_dim, self.activation).to(self.device)
        self.reset_optimizer_scheduler() # do not delete this


    def reset_optimizer_scheduler(self,):
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)
        # self.scheduler = CosineAnnealingLR(default_optimizer, 1000, 1e-6)


    def train(self, train_loader):
        losses = []
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.to(self.device)
            data = data.to(self.device)

            self.optimizer.zero_grad()
            output_ = []
            kl_ = []
            for mc_run in range(self.num_mc):
                output, kl = self.model(data)
                output_.append(output)
                kl_.append(kl)
            output = torch.mean(torch.stack(output_), dim=0)
            kl = torch.mean(torch.stack(kl_), dim=0)
            nll_loss = F.mse_loss(output, target)
            #ELBO loss
            loss = nll_loss + (kl / self.batch_size)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        # print('loss: {:.4f}, lr: {:.4f}'.format(np.mean(losses), self.optimizer.param_groups[0]['lr']), end=',')

        return losses


    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = target.to(self.device)
                data = data.to(self.device)

                output, kl = self.model(data)
                test_loss += F.mse_loss(output, target, reduction='sum').item() + (
                    kl / self.batch_size)  # sum up batch loss

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}\n'.format(test_loss))


    def evaluate(self, test_loader):
        test_loss = []
        
        with torch.no_grad():
            for data, target in test_loader:
                target = target.to(self.device)
                data = data.to(self.device)

                predicts = []
                for mc_run in range(self.num_monte_carlo):
                    self.model.eval()
                    output, _ = self.model.forward(data)
                    loss = F.mse_loss(output, target).cpu().data.numpy()
                    test_loss.append(loss)
                    predicts.append(output.cpu().data.numpy())

                pred_mean = np.mean(predicts, axis=0)
                pred_var = np.var(predicts, axis=0)

                print('prediction mean: ',pred_mean, 'prediction var: ', pred_var)
            
            print('test loss: ', np.mean(test_loss))


def import_data(file):
    import pickle
    results = pickle.load(open(file, 'rb'))
    inputs, outputs = [], []
    for r in results:
        act = r[1]
        inputs.append(np.asarray([act[key] for key in act.keys()]))
        outputs.append(r[3])

    return np.array(inputs), np.array(outputs)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch simple_bnn Example')
    parser.add_argument('--inputdim',
                        type=int,
                        default=7,
                        metavar='N',
                        help='input dim size for training (default: 14)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1.0,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.999,
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
        default='./logs/main_bnn',
        metavar='N',
        help=
        'use tensorboard for logging and visualization of training progress')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ############################################################################################################
    inputs = np.random.random((1000, args.inputdim))
    outputs = np.sum(inputs, axis=-1)

    inputs, outputs = import_data("./dataset/dataset.pkl")

    inputs = inputs.astype(np.float32)
    outputs = np.squeeze(-outputs).astype(np.float32)

    train_loader = torch.utils.data.DataLoader(Dataset(inputs, outputs), batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(Dataset(inputs, outputs), batch_size=args.batch_size, shuffle=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ############################################################################################################
    bnn = BNN(input_dim=inputs.shape[-1],batch_size=args.batch_size, lr=args.lr, gamma=args.gamma)

    print(args.mode)
    if args.mode == 'train':
        losses = []
        for epoch in range(args.epochs):
            print("epoch " + str(epoch), end=', ')
            loss = bnn.train(train_loader)
            bnn.scheduler.step()
            bnn.test(test_loader)
            losses += loss
            if epoch % 10 == 0:        
                torch.save(bnn.model.state_dict(), args.save_dir + "/simple_bnn_bayesian_fc.pth")

        plt.plot(losses)
        plt.ylim(0, 10)
        plt.savefig("figures/result_bnn_training_test.png", format = 'png', dpi=300)

    elif args.mode == 'test':
        checkpoint = args.save_dir + '/simple_bnn_bayesian_fc.pth'
        bnn.model.load_state_dict(torch.load(checkpoint))
        bnn.evaluate(train_loader)
        bnn.evaluate(test_loader)

    print("done.")


if __name__ == '__main__':
    main()
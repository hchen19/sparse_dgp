'''
Bayesian Residual DGP for CIFAR10.

ResNet architecture ref:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from sparse_dgp.layers import LinearReparameterization
from sparse_dgp.layers import AMGP

__all__ = [
    'ResNet', 'resnet8', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110'
]


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, features, design_class, kernel, n_level=5, option='A'):
        super(BasicBlock, self).__init__()
        self.gp1 = AMGP(
            in_features = in_features,
            n_level = n_level,
            design_class = design_class,
            kernel = kernel
        )
        self.fc1 = LinearReparameterization(
            in_features = self.gp1.out_features,
            out_features = features,
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
            bias = False
        )
        self.gp2 = AMGP(
            in_features = features,
            n_level = n_level,
            design_class = design_class,
            kernel = kernel
        )
        self.fc2 = LinearReparameterization(
            in_features = self.gp2.out_features,
            out_features = features,
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
            bias = False
        )

        self.shortcut = nn.Sequential()
        if in_features != features:
            padding_size = features - in_features
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x,
                    (padding_size // 2, padding_size // 2), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    LinearReparameterization(
                        in_features = in_features,
                        out_features = features,
                        prior_mean = prior_mu,
                        prior_variance = prior_sigma,
                        posterior_mu_init = posterior_mu_init,
                        posterior_rho_init = posterior_rho_init,
                        bias = False
                    ), nn.LayerNorm(features)
                )

    def forward(self, x):
        kl_sum = 0
        x = self.gp1(x)
        x, kl = self.fc1(x)
        kl_sum += kl
        x = self.gp2(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        x += self.shortcut(x)

        return x, kl_sum


class ResNet(nn.Module):
    def __init__(self, input_dim, design_class, kernel, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_features = 64

        self.fc0 = LinearReparameterization(
            in_features=input_dim,
            out_features=64,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        self.layer1 = self._make_layer(design_class, kernel, block, 64, num_blocks[0])
        self.layer2 = self._make_layer(design_class, kernel, block, 128, num_blocks[1])
        self.layer3 = self._make_layer(design_class, kernel, block, 32, num_blocks[2])
        self.gp1 = AMGP(in_features = 32, n_level = 5, design_class = design_class, kernel = kernel)
        self.fc1 = LinearReparameterization(
            in_features = self.gp1.out_features,
            out_features = num_classes,
            prior_mean = prior_mu,
            prior_variance = prior_sigma,
            posterior_mu_init = posterior_mu_init,
            posterior_rho_init = posterior_rho_init,
            bias = True
        )

    def _make_layer(self, design_class, kernel, block, features, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.in_features, features, design_class, kernel))
            self.in_features = features
        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        x = self.fc0(x)
        for layer in self.layer1:
            x, kl = layer(x)
            kl_sum += kl
        for layer in self.layer2:
            x, kl = layer(x)
            kl_sum += kl
        for layer in self.layer3:
            x, kl = layer(x)
            kl_sum += kl
        x = self.gp1(x)
        x, kl = self.fc1(x)
        kl_sum += kl
        return x, kl_sum


def resnet8(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [1, 1, 1], num_classes)

def resnet20(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [3, 3, 3], num_classes)

def resnet32(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [5, 5, 5], num_classes)

def resnet44(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [7, 7, 7], num_classes)

def resnet56(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [9, 9, 9], num_classes)

def resnet110(input_dim, design_class, kernel, num_classes):
    return ResNet(input_dim, design_class, kernel, BasicBlock, [18, 18, 18], num_classes)



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtmgp.layers.linear_variational import LinearReparameterization
from dtmgp.layers.tmk import TMK


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class SimpleDTMGP(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 kernel, sparse_grid, chol_inv,
                 num_hidden_layers=2, 
                 activation=None):
        super(SimpleDTMGP, self).__init__()

        self.activation = activation
        self.tmk = TMK(kernel=kernel, sparse_grid=sparse_grid, chol_inv=chol_inv)

        self.fc1 = LinearReparameterization(
            in_features=input_dim,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterization(
            in_features=128,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc3 = LinearReparameterization(
            in_features=256,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc4 = LinearReparameterization(
            in_features=128,
            out_features=output_dim,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.hidden = LinearReparameterization(
            in_features=128,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        # layers = [self.fc1, TMK()]
        # for i in range(num_hidden_layers):
        #     layers.append(self.hidden)
        #     layers.append(TMK())
        # layers.append(self.fc4)
        # net_x = nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0
        x, kl = self.fc1(x)
        kl_sum += kl
        x = self.tmk(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        x = self.tmk(x)
        x, kl = self.fc3(x)
        kl_sum += kl
        x = self.tmk(x) 
        x, kl = self.fc4(x)
        kl_sum += kl
        if self.activation is None:
            output = x
        else:
            output = self.activation(x) # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output), kl_sum 
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtmgp.layers.linear_variational import LinearReparameterization
from dtmgp.layers.tmgps import tmgp_additive

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class AdditiveDTMGP(nn.Module):
    def __init__(self, input_dim, output_dim,
                 design_class, kernel,
                 activation=None):
        super(AdditiveDTMGP, self).__init__()

        self.activation = activation

        #################################################################################
        ## 1st layer of DGP: input:[...,n, input_dim] size tensor, output:[...,n, w1] size tensor
        #################################################################################
        # return [input_dim,...,n, m1] size tensor for [...,n, input_dim] size input and [m1, 1] size design points
        self.tmk1 = tmgp_additive(in_features=input_dim, n_level=5, design_class=design_class, kernel=kernel)        
        m1 = self.tmk1.out_features
        w1 = 8
        # return [input_dim,...,n, w1] size tensor for [input_dim,...,n, m1] size input and [m1, w1] size weights
        self.fc1 = LinearReparameterization(
            in_features=m1,
            out_features=w1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # move axis of input: [input_dim,...,n, w1] --> [...,n, w1, input_dim], use `x.moveaxis(0,-1)`
        # return [...,n, w1,1] size tensor for [...,n, w1, input_dim] size input and [input_dim, 1] size weights
        self.fc_add1 = LinearReparameterization(
            in_features=input_dim,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # squeeze input: [...,n, w1,1] --> [...,n, w1], use `x.squeeze(dim=-1)`


        #################################################################################
        ## 2nd layer of DGP: input:[...,n, w1] size tensor, output:[...,n, w2] size tensor
        #################################################################################
        # return [w1,..., n, m2] size tensor for [..., n, w1] size input and [m2, 1] size design points
        self.tmk2 = tmgp_additive(in_features=w1, n_level=6, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.out_features
        w2 = 10
        # return [w1,...,n, w2] size tensor for [w1,..., n, m2] size input and [m2, w2] size weights
        self.fc2 = LinearReparameterization(
            in_features=m2,
            out_features=w2,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # move axis of input: [w1,...,n, w2] --> [...,n, w2, w1], use `x.moveaxis(0,-1)`
        # return [...,n, w2,1] size tensor for [...,n, w2, w1] size input and [w1, 1] size weights
        self.fc_add2 = LinearReparameterization(
            in_features=w1,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # squeeze input: [...,n, w2,1] --> [...,n, w2], use `x.squeeze(dim=-1)`


        #################################################################################
        ## 3rd layer of DGP: input:[...,n, w2] size tensor, output:[...,n, w3] size tensor
        #################################################################################
        # return [w2,...,n, m3] size tensor for [...,n, w2] size input and [m3, 1] size design points
        self.tmk3 = tmgp_additive(in_features=w2, n_level=6, design_class=design_class, kernel=kernel)
        m3 = self.tmk3.out_features
        w3 = output_dim
        # return [w2,...,n, w3] size tensor for [w2,...,n, m3] size input and [m3, w3] size weights
        self.fc3 = LinearReparameterization(
            in_features=m3,
            out_features=w3,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # move axis of input: [w2,...,n, w3] --> [...,n, w3, w2], use `x.moveaxis(0,-1)`
        # return [...,n, w3,1] size tensor for [...,n, w3, w2] size input and [w2, 1] size weights
        self.fc_add3 = LinearReparameterization(
            in_features=w2,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )
        # squeeze input: [...,n, w3,1] --> [...,n, w3], use `x.squeeze(dim=-1)`
    
    
    def forward(self, x):
        kl_sum = 0
        
        x = self.tmk1(x)
        x, _ = self.fc1(x)
        x, kl = self.fc_add1(x.moveaxis(0,-1))
        x = x.squeeze(dim=-1)
        kl_sum += kl.squeeze(dim=-1)

        x = self.tmk2(x)
        x, _ = self.fc2(x)
        x, kl = self.fc_add2(x.moveaxis(0,-1))
        x = x.squeeze(dim=-1)
        kl_sum += kl.squeeze(dim=-1)

        x = self.tmk3(x)
        x, _ = self.fc3(x)
        x, kl = self.fc_add3(x.moveaxis(0,-1))
        x = x.squeeze(dim=-1)
        kl_sum += kl.squeeze(dim=-1)

        if self.activation is None:
            output = x
        else:
            output = self.activation(x)  # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output), kl_sum

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from dtmgp.layers.linear_variational import LinearReparameterization
from dtmgp.layers.tmgp_variational import TMK


prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class SimpleDTMGP(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 design_class, kernel, 
                 activation=None):
        super(SimpleDTMGP, self).__init__()

        self.activation = activation

        ##########################
        ## 1st layer of DGP
        ##########################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid 
        self.tmk1 = TMK(feature_dim=input_dim, n_level=2, design_class=design_class, kernel=kernel)
        m1 = self.tmk1.n_points

        # return [n, 256] size tensor for [n, m1] size input and [m1, 256] size weights
        self.fc1 = LinearReparameterization(
            in_features=m1,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        ##########################
        ## 2nd layer of DGP
        ##########################
        # return [n, 256, m2] size tensor for [n, 256, 1] size input and [m2, 1] size sparse grid
        # or return [n, m2] size tensor for [n, 256] size input and [m2, 256] size sparse grid
        self.tmk2 = TMK(feature_dim=256, n_level=2, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.n_points

        # return [n, 256, 512] size tensor for [n, 256, m2] size input and [m2, 512] size weights
        # or return [n, 512] size tensor for [n, 256, m2] size input and [256, m2, 512] size weights
        # or return [n, 512] size tensor for [n, m2] size input and [m2, 512] size weights
        self.fc2 = LinearReparameterization(
            in_features=m2,
            out_features=512,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        
        ##########################
        ## 3rd layer of DGP
        ##########################
        # return [n, 256, 512, m3] size tensor for [n, 256, 512, 1] size input and [m3, 1] size sparse grid
        # return [n, 512, m3] size tensor for [n, 512, 1] size input and [m3, 1] size sparse grid
        # or return [n, m3] size tensor for [n, 512] size input and [m3, 512] size sparse grid
        self.tmk3 = TMK(feature_dim=512, n_level=2, design_class=design_class, kernel=kernel)
        m3 = self.tmk3.n_points

        # return [n, 256, 512, 1] size tensor for [n, 256, 512, m3] size input and [m3, 1] size weights
        # or return [n, 1] size tensor for [n, 512, m3] size input and [512, m3, 1] size weights
        # or return [n, 1] size tensor for [n, m3] size input and [m3, 1] size weights
        self.fc3 = LinearReparameterization(
            in_features=m3,
            out_features=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )


        self.fc4 = LinearReparameterization(
            in_features=241,
            out_features=output_dim,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        # self.hidden = LinearReparameterization(
        #     in_features=128,
        #     out_features=128,
        #     prior_mean=prior_mu,
        #     prior_variance=prior_sigma,
        #     posterior_mu_init=posterior_mu_init,
        #     posterior_rho_init=posterior_rho_init,
        # )

        # layers = [self.fc1, TMK()]
        # for i in range(num_hidden_layers):
        #     layers.append(self.hidden)
        #     layers.append(TMK())
        # layers.append(self.fc4)
        # net_x = nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0

        x = self.tmk1(x)
        x, kl = self.fc1(x)
        kl_sum += kl
        
        x = self.tmk2(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        
        x = self.tmk3(x)
        x, kl = self.fc3(x)
        kl_sum += kl

        # x, kl = self.fc4(x)
        # kl_sum += kl
        if self.activation is None:
            output = x
        else:
            output = self.activation(x) # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output), kl_sum 
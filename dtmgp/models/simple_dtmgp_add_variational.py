from __future__ import print_function
import torch
import torch.nn as nn

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
        ## 1st layer of DGP: input:[n, input_dim] size tensor, output:[n, w1] size tensor
        #################################################################################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid
        self.tmk1 = tmgp_additive(in_features=input_dim, n_level=5, design_class=design_class, kernel=kernel)
        m1 = self.tmk1.out_features * input_dim
        w1 = 8
        # return [n, w1] size tensor for [n, m1] size input and [m1, w1] size weights
        self.fc1 = LinearReparameterization(
            in_features=m1,
            out_features=w1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 2nd layer of DGP: input:[n, w1] size tensor, output:[n, w2] size tensor
        #################################################################################
        # return [n, m2] size tensor for [n, w1] size input and [m2, w1] size sparse grid
        self.tmk2 = tmgp_additive(in_features=w1, n_level=5, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.out_features * w1
        w2 = 8
        # return [n, w2] size tensor for [n, m2] size input and [m2, w2] size weights
        self.fc2 = LinearReparameterization(
            in_features=m2,
            out_features=w2,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 3rd layer of DGP: input:[n, w2] size tensor, output:[n, w3] size tensor
        #################################################################################
        # return [n, m3] size tensor for [n, w2] size input and [m3, w2] size sparse grid
        self.tmk3 = tmgp_additive(in_features=w2, n_level=5, design_class=design_class, kernel=kernel)
        m3 = self.tmk3.out_features * w2
        # return [n, w3] size tensor for [n, m3] size input and [m3, w3] size weights
        self.fc3 = LinearReparameterization(
            in_features=m3,
            out_features=output_dim,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

    def forward(self, x):
        kl_sum = 0

        x = self.tmk1(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc1(x)
        kl_sum += kl

        x = self.tmk2(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc2(x)
        kl_sum += kl

        x = self.tmk3(x)
        x = torch.flatten(x, 1)
        x, kl = self.fc3(x)
        kl_sum += kl

        if self.activation is None:
            output = x
        else:
            output = self.activation(x)  # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output), kl_sum

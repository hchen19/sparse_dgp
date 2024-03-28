from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

from dtmgp.layers import LinearReparameterization
from dtmgp.layers import AdditiveTMGP

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class DTMGPmnist(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 design_class, 
                 kernel,
                 activation=None):
        super(DTMGPmnist, self).__init__()

        self.activation = activation

        #################################################################################
        ## 1st layer of DGP: input:[n, input_dim] size tensor, output:[n, w1] size tensor
        #################################################################################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid
        self.tmk1 = AdditiveTMGP(in_features=input_dim, n_level=3, design_class=design_class, kernel=kernel)
        m1 = self.tmk1.out_features # m1 = input_dim*(2^n_level-1)
        w1 = 512
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
        self.tmk2 = AdditiveTMGP(in_features=w1, n_level=4, design_class=design_class, kernel=kernel)
        m2 = self.tmk2.out_features # m2 = w1*(2^n_level-1)
        w2 = 128
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
        self.tmk3 = AdditiveTMGP(in_features=w2, n_level=6, design_class=design_class, kernel=kernel)
        W3 = 32
        m3 = self.tmk3.out_features # m3 = w2*(2^n_level-1)
        # return [n, w3] size tensor for [n, m3] size input and [m3, w3] size weights
        self.fc3 = LinearReparameterization(
            in_features=m3,
            out_features=W3,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 4th layer of DGP: input:[n, w3] size tensor, output:[n, w4] size tensor
        #################################################################################
        # return [n, m4] size tensor for [n, w3] size input and [m4, w4] size sparse grid
        self.tmk4 = AdditiveTMGP(in_features=W3, n_level=6, design_class=design_class, kernel=kernel)
        m4 = self.tmk4.out_features # m3 = w2*(2^n_level-1)
        # return [n, w3] size tensor for [n, m3] size input and [m3, w3] size weights
        self.fc4 = LinearReparameterization(
            in_features=m4,
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
        x, kl = self.fc1(x)
        kl_sum += kl

        x = self.tmk2(x)
        x, kl = self.fc2(x)
        kl_sum += kl

        x = self.tmk3(x)
        x, kl = self.fc3(x)
        kl_sum += kl

        x = self.tmk4(x)
        x, kl = self.fc4(x)
        kl_sum += kl

        if self.activation is None:
            output = F.log_softmax(x, dim=1)
        else:
            output = self.activation(x)  # attention, this only regress non-negative values TODO XXX
        return output, kl_sum

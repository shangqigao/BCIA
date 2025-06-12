import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Basic_module import Visualization
from .ResNet import ResNet_appearance, ResNet_shape
from .build import register_decomposition

logger = logging.getLogger(__name__)

class BayesDec(nn.Module):
    def __init__(self, cfg):
        """Bayesian image decomposition: image = shape + appearance
        """
        super(BayesDec, self).__init__()

        self.in_channel = cfg['MODEL']['DECOMPOSITION']['IN_CHANNEL']
        self.mu_0 = cfg['MODEL']['DECOMPOSITION']['MU_0']
        self.sigma_0 = cfg['MODEL']['DECOMPOSITION']['SIGMA_0']
        self.gamma_rho = cfg['MODEL']['DECOMPOSITION']['GAMMA_RHO']
        self.phi_rho = float(cfg['MODEL']['DECOMPOSITION']['PHI_RHO'])
        self.gamma_upsilon = cfg['MODEL']['DECOMPOSITION']['GAMMA_UPSILON']
        self.phi_upsilon = float(cfg['MODEL']['DECOMPOSITION']['PHI_UPSILON'])

        self.res_shape = ResNet_shape(num_in_ch=self.in_channel, num_out_ch=2*self.in_channel)
        self.res_appear = ResNet_appearance(num_in_ch=self.in_channel, num_out_ch=2*self.in_channel, num_block=6, bn=True)
        self.group_norm = nn.GroupNorm(self.in_channel, self.in_channel)

        Dx = torch.zeros([1, 1, 3, 3], dtype=torch.float)
        Dx[:, :, 1, 1] = 1
        Dx[:, :, 1, 0] = Dx[:, :, 1, 2] = Dx[:, :, 0, 1] = Dx[:, :, 2, 1] = -1 / 4
        # repeat for multi-channel image
        Dx_grouped = Dx.repeat(self.in_channel, 1, 1, 1)
        self.Dx = nn.Parameter(data=Dx_grouped, requires_grad=False)


    @staticmethod
    def sample_normal_jit(mu, log_var):
        sigma = torch.exp(log_var / 2)
        eps = mu.mul(0).normal_()
        z = eps.mul_(sigma).add_(mu)
        return z, eps

    def generate_m(self, samples):
        feature = self.res_appear(samples)
        mu_m, log_var_m = torch.chunk(feature, 2, dim=1)
        log_var_m = torch.clamp(log_var_m, -20, 0)
        m, _ = self.sample_normal_jit(mu_m, log_var_m)
        return m, mu_m, log_var_m

    def generate_x(self, samples):
        feature = self.res_shape(samples)
        mu_x, log_var_x = torch.chunk(feature, 2, dim=1)
        log_var_x = torch.clamp(log_var_x, -20, 0)
        x, _ = self.sample_normal_jit(mu_x, log_var_x)
        return x, mu_x, log_var_x

    def forward(self, samples: torch.Tensor):
        x, mu_x, log_var_x = self.generate_x(samples)
        m, mu_m, log_var_m = self.generate_m(samples)

        residual = samples - (x + m)
        mu_rho_hat = (2 * self.gamma_rho + 1) / (
            residual * residual + 2 * self.phi_rho
        )
        # mu_rho_hat = torch.clamp(mu_rho_hat, 1e4, 1e8)

        normalization = torch.sum(mu_rho_hat).detach()
        n, _ = self.sample_normal_jit(m, torch.log(1 / mu_rho_hat))

        # Image line upsilon
        alpha_upsilon_hat = 2 * self.gamma_upsilon + self.in_channel
        difference_x = F.conv2d(mu_x, self.Dx, padding=1, groups=self.in_channel)
        beta_upsilon_hat = (
            torch.sum(
                difference_x * difference_x + 2 * torch.exp(log_var_x),
                dim=1,
                keepdim=True,
            )
            + 2 * self.phi_upsilon
        )  # B x 1 x W x H
        mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
        # mu_upsilon_hat = torch.clamp(mu_upsilon_hat, 1e6, 1e10)

        # compute loss-related
        kl_y = residual * mu_rho_hat.detach() * residual

        kl_mu_x = torch.sum(
            difference_x * difference_x * mu_upsilon_hat.detach(), dim=1
        )
        kl_sigma_x = torch.sum(
           2 * torch.exp(log_var_x) * mu_upsilon_hat.detach() - log_var_x,
           dim=1
        )

        kl_mu_m = self.sigma_0 * (mu_m - self.mu_0) * (mu_m - self.mu_0)
        kl_sigma_m = self.sigma_0 * torch.exp(log_var_m) - log_var_m

        visualize = {
            "shape": torch.concat([x, mu_x, torch.exp(log_var_x / 2)]),
            "appearance": torch.concat([n, m, 1 / mu_rho_hat.sqrt()]),
            "shape_boundary": mu_upsilon_hat,
        }

        pred = x if self.training else mu_x
        pred = self.group_norm(pred)
        out = {
            "pred": pred,
            "kl_y": kl_y,
            "kl_mu_x": kl_mu_x,
            "kl_sigma_x": kl_sigma_x,
            "kl_mu_m": kl_mu_m,
            "kl_sigma_m": kl_sigma_m,
            "normalization": normalization,
            "rho": mu_rho_hat,
            "upsilon": mu_upsilon_hat,
            "visualize": visualize,
        }
        return out

class BayesDecVis(Visualization):
    def __init__(self):
        super(BayesDecVis, self).__init__()

    def forward(self, inputs, outputs, step, writer):
        self.save_image(inputs, "inputs", step, writer)
        for key, value in outputs.items():
            self.save_image(value, key, step, writer)


def build_visualizer():
    return BayesDecVis() 

@register_decomposition
def get_bayes_decomposition(cfg, **kwargs):
    return BayesDec(cfg) 

import torch

from src import config
from src.models import SNGANGModel, SNGANDModel
from ._gan import GAN


class SNGAN(GAN):
    def __init__(self):
        super().__init__(SNGANGModel(), SNGANDModel())

    def _fit(self, x: torch.Tensor):
        d_optimizer = torch.optim.Adam(
            params=self.d.parameters(),
            lr=config.gan_config.d_lr,
            betas=(0.5, 0.999),
        )
        g_optimizer = torch.optim.Adam(
            params=self.g.parameters(),
            lr=config.gan_config.g_lr,
            betas=(0.5, 0.999),
        )

        for _ in range(config.gan_config.epochs):
            for __ in range(config.gan_config.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
            for __ in range(config.gan_config.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()

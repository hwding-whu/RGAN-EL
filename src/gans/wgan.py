import torch

from src import config
from src.models import WGANGModel, WGANDModel
from src.gans._gan import GAN


class WGAN(GAN):
    def __init__(self):
        super().__init__(WGANGModel(), WGANDModel())

    def _fit(self, x: torch.Tensor):
        d_optimizer = torch.optim.RMSprop(
            params=self.d.parameters(),
            lr=config.gan_config.d_lr
        )
        g_optimizer = torch.optim.RMSprop(
            params=self.g.parameters(),
            lr=config.gan_config.g_lr,
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
                for p in self.d.parameters():
                    p.data.clamp_(*config.gan_config.wgan_clamp)
            for __ in range(config.gan_config.g_loops):
                self.g.zero_grad()
                z = torch.randn(len(x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z)
                prediction = self.d(fake_x)
                loss = - prediction.mean()
                loss.backward()
                g_optimizer.step()

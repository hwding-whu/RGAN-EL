import torch
from torch.nn.functional import binary_cross_entropy

from src import config
from src.models import WGANGModel, WGANDModel
from src.r_gans._r_gan import RGAN


class RWGAN(RGAN):
    def __init__(self):
        super().__init__(WGANGModel(), WGANDModel())

    def _fit(
            self,
            pos_x: torch.Tensor,
            neg_x: torch.Tensor,
    ):
        d_optimizer = torch.optim.RMSprop(
            params=self.d.parameters(),
            lr=config.gan_config.d_lr
        )
        g_optimizer = torch.optim.RMSprop(
            params=self.g.parameters(),
            lr=config.gan_config.g_lr,
        )
        c_optimizer = torch.optim.Adam(
            params=self.c.parameters(),
            lr=1e-3,
        )
        for _ in range(config.gan_config.epochs):
            for __ in range(config.gan_config.d_loops):
                self.d.zero_grad()
                prediction_real = self.d(pos_x)
                loss_real = - prediction_real.mean()
                z = torch.randn(len(pos_x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z).detach()
                prediction_fake = self.d(fake_x)
                loss_fake = prediction_fake.mean()
                loss = loss_real + loss_fake
                loss.backward()
                d_optimizer.step()
                for p in self.d.parameters():
                    p.data.clamp_(*config.gan_config.wgan_clamp)
            for __ in range(3):
                self.c.zero_grad()
                x = torch.cat([pos_x, neg_x])
                labels = torch.cat([
                    torch.ones(len(pos_x), device=config.device),
                    torch.zeros(len(neg_x), device=config.device),
                ])
                prediction = self.c(x).squeeze()
                loss = binary_cross_entropy(
                    input=prediction,
                    target=labels,
                )
                loss.backward()
                c_optimizer.step()
            for __ in range(config.gan_config.g_loops):
                self.g.zero_grad()

                self.c(neg_x)
                c_neg_hidden_output = self.c.hidden_output.detach()
                z = torch.randn(len(neg_x), config.model_config.z_size, device=config.device)
                fake_pos_x = self.g(z)
                self.c(fake_pos_x)
                c_pos_hidden_output = self.c.hidden_output.detach()
                c_hidden_loss = torch.norm(c_neg_hidden_output - c_pos_hidden_output, p=2)

                self.d(pos_x)
                z = torch.randn(len(pos_x), config.model_config.z_size, device=config.device)
                fake_x = self.g(z)
                d_real_x_hidden_output = self.d.hidden_output.detach()
                d_final_output = self.d(fake_x)
                d_fake_x_hidden_output = self.d.hidden_output
                d_hidden_loss = torch.norm(d_real_x_hidden_output - d_fake_x_hidden_output, p=2)

                loss = -d_final_output.mean() + d_hidden_loss - c_hidden_loss
                loss.backward()
                g_optimizer.step()

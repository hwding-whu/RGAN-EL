import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from src import models
from src.models._model import Model


class SNGANDModel(Model):
    def __init__(self):
        super().__init__()
        self.main_model = nn.Sequential(
            spectral_norm(nn.Linear(models.x_size, 512)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(512, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 32)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(32, 8)),
            nn.LeakyReLU(0.2),
        )
        self.last_layer = spectral_norm(nn.Linear(8, 1))
        self.hidden_output = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_output = self.main_model(x)
        return self.last_layer(self.hidden_output)

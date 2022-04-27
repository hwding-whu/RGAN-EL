import torch
from torch import nn

from src import models
from src.models._model import Model


class WGANDModel(Model):
    def __init__(self):
        super().__init__()
        self.main_model = nn.Sequential(
            nn.Linear(models.x_size, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.2),

        )
        self.last_layer = nn.Linear(8, 1)
        self.hidden_output = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_output = self.main_model(x)
        return self.last_layer(
            self.hidden_output
        )

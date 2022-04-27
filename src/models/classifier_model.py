import torch
from torch import nn

from src import models
from src.models._model import Model


class ClassifierModel(Model):
    def __init__(self):
        super().__init__()
        self.main_model = nn.Sequential(
            nn.Linear(models.x_size, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 16),
        )
        self.last_layer = nn.Linear(16, 1)
        self.hidden_output = None

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        self.hidden_output = self.main_model(x)
        x = self.last_layer(self.hidden_output)
        return torch.sigmoid(x)

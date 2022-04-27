from abc import abstractmethod

import torch

from src import config
from src.logger import Logger
from src.models import Model
from src.datasets import Dataset


class GAN:
    def __init__(
            self,
            g: Model,
            d: Model,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g.to(config.device)
        self.d = d.to(config.device)

    def fit(self, dataset: Dataset):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        x = dataset.to(config.device).get_samples()
        self._fit(x)
        self.g.eval()
        self.d.eval()
        self.logger.info(f'Finished training')

    @abstractmethod
    def _fit(self, x: torch.Tensor):
        pass

    def generate_samples(self, size: int):
        z = torch.randn(size, config.model_config.z_size, device=config.device)
        return self.g(z).detach()

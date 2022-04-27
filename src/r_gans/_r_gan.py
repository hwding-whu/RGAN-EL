import torch

from src import config
from src.logger import Logger
from src.models import Model, ClassifierModel
from src.datasets import Dataset, NegativeDataset


class RGAN:
    def __init__(
            self,
            g: Model,
            d: Model,
    ):
        self.logger = Logger(self.__class__.__name__)
        self.g = g.to(config.device)
        self.d = d.to(config.device)
        self.c = ClassifierModel().to(config.device)

    def fit(
            self,
            dataset: Dataset,
    ):
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')
        pos_x = dataset.to(config.device).get_samples()
        neg_x = NegativeDataset().to(config.device).get_samples()
        self._fit(pos_x, neg_x)
        self.g.eval()
        self.d.eval()
        self.c.eval()
        self.logger.info(f'Finished training')

    def _fit(
            self,
            pos_x: torch.Tensor,
            neg_x: torch.Tensor,
    ):
        raise NotImplementedError

    def _generate_sample(self):
        z = torch.randn(1, config.model_config.z_size, device=config.device)
        return self.g(z).detach()

    def generate_samples(self, size: int):
        samples = []
        cnt = 0
        patience = 100
        while cnt < size:
            x = self._generate_sample()
            prediction = self.c(x)
            if prediction.item() > 0.5 or patience == 0:
                samples.append(x)
                cnt += 1
                patience = 100
            else:
                patience -= 1
        return torch.cat(samples)

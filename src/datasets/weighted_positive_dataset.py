import random

import torch
import numpy as np

from src.datasets import PositiveDataset, NegativeDataset


class WeightedPositiveDataset(PositiveDataset):
    def __init__(
            self,
            test: bool = False,
    ):
        super().__init__(test)
        self.is_weighted = True
        # calculate weights
        pos_samples = PositiveDataset(test).samples
        neg_samples = NegativeDataset(test).samples
        dist = np.zeros([len(pos_samples), len(neg_samples)])

        # calculate distances
        for i, minority_item in enumerate(pos_samples):
            for j, majority_item in enumerate(neg_samples):
                dist[i][j] = torch.norm(minority_item - majority_item, p=2) + 1e-3

        self.fits = 1 / dist.sum(axis=1, initial=None)
        self.weights = self.fits / self.fits.sum()

    def _get_weighted_samples(self, size: int) -> torch.Tensor:
        return torch.stack(
            random.choices(
                self.samples,
                weights=self.weights,
                k=size,
            )
        )

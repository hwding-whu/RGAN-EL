from typing import Callable

import torch

from src import datasets
from src.datasets._dataset import Dataset


class FullDataset(Dataset):
    def __init__(self, test: bool = False):
        super().__init__(test)
        self.samples = datasets.test_samples if test else datasets.training_samples
        self.samples = torch.from_numpy(self.samples).float()
        self.labels = datasets.test_labels if test else datasets.training_labels
        self.labels = torch.from_numpy(self.labels).float()

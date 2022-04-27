from __future__ import annotations

import torch


class Dataset:
    def __init__(
            self,
            test: bool = False,
    ):
        self.samples: torch.Tensor = None
        self.labels: torch.Tensor = None
        self.test = test
        self.is_weighted = False

    def __len__(self):
        return len(self.labels)

    def to(self, device: str) -> Dataset:
        self.samples = self.samples.to(device)
        self.labels = self.labels.to(device)
        return self

    def get_samples(self, size: int = None) -> torch.Tensor:
        if size is None:
            size = len(self)
        if self.is_weighted:
            return self._get_weighted_samples(size)
        else:
            return self.samples[:size]

    def _get_weighted_samples(self, size: int) -> torch.Tensor:
        pass

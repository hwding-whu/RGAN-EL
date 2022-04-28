from math import ceil

import torch
from imblearn.over_sampling import SMOTE

import src


class EPipe:
    def __init__(self):
        pass

    def get_datasets(
            self,
            over_sampler=SMOTE(random_state=src.config.seed),
    ):
        all_negative_samples = src.datasets.NegativeDataset().samples
        all_positive_samples = src.datasets.PositiveDataset().samples
        result = []
        nn, no = self._divide_samples(all_negative_samples, all_positive_samples)
        nns = nn.split(ceil(len(nn) / 3))
        for nn in nns:
            n = torch.cat([nn, no], dim=0)
            p = all_positive_samples
            x = torch.cat([n, p])
            y = torch.cat(
                [
                    torch.zeros(len(n)),
                    torch.ones(len(p)),
                ]
            )
            x = x.cpu().numpy()
            y = y.cpu().numpy()
            x, y = over_sampler.fit_resample(x, y)
            balanced_dataset = src.datasets.Dataset()
            balanced_dataset.samples = torch.from_numpy(x)
            balanced_dataset.labels = torch.from_numpy(y)
            result.append(balanced_dataset)
        return result

    @staticmethod
    def _divide_samples(
            neg_samples: torch.Tensor,
            pos_samples: torch.Tensor,
    ):
        dists = torch.zeros([len(pos_samples) + len(neg_samples)] * 2)
        all_samples = torch.cat([pos_samples, neg_samples])
        for i, sample_a in enumerate(all_samples):
            for j, sample_b in enumerate(all_samples):
                if i > j:
                    dists[i][j] = dists[j][i]
                elif i < j:
                    dists[i][j] = torch.norm(sample_a - sample_b, p=2)
                else:
                    continue
        all_labels = torch.cat(
            [
                torch.ones(len(pos_samples)),
                torch.zeros(len(neg_samples))
            ]
        )
        k = 3
        nn, no = [], []
        for i, sample in enumerate(neg_samples):
            indices = torch.topk(dists[i], k + 1, largest=False).indices[1:]
            labels = all_labels[indices]
            p = sum(labels) / k - 1e-5
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            if entropy > 0.5:
                no.append(sample)
            else:
                nn.append(sample)
        if nn:
            nn = torch.stack(nn, dim=0)
        else:
            nn = torch.tensor([])
        if no:
            no = torch.stack(no, dim=0)
        else:
            no = torch.tensor([])
        return nn, no


class EntropyVoter(src.voter.Voter):
    def __init__(self):
        super().__init__()
        self.classifiers = [src.classifier.Classifier('V') for _ in range(3)]

    def fit(self, over_sampler=SMOTE(random_state=src.config.seed)) -> None:
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {src.config.device}')
        datasets = EPipe().get_datasets(over_sampler)
        for classifier, dataset in zip(self.classifiers, datasets):
            classifier.fit(dataset)

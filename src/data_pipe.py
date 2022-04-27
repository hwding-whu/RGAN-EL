import typing
import math

import torch
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE

import src


class DataPipe:
    def __init__(self):
        pass

    def get_datasets(
            self,
            over_sampler: typing.Union[
                src.types.GAN,
                src.types.RGAN,
                SMOTE,
                ADASYN,
                SVMSMOTE,
            ] = None,
    ) -> typing.List[src.types.Dataset]:
        all_negative_samples = src.datasets.NegativeDataset().samples
        all_positive_samples = src.datasets.PositiveDataset().samples
        result = []
        if over_sampler is None:
            ns = all_negative_samples.split(math.ceil(len(all_negative_samples) / 3))
            ps = all_positive_samples.split(math.ceil(len(all_positive_samples) / 3))
            for n, p in zip(ns, ps):
                dataset = src.datasets.Dataset()
                dataset.samples = torch.cat([n, p], dim=0)
                dataset.labels = torch.cat(
                    [
                        torch.zeros(len(n)),
                        torch.ones(len(p))
                    ]
                )
                result.append(dataset)
        elif issubclass(over_sampler.__class__, src.types.GAN) or issubclass(over_sampler.__class__, src.types.RGAN):
            nn, no = self._divide_samples(all_negative_samples, all_positive_samples)
            nns = nn.split(math.ceil(len(nn) / 3))
            for nn in nns:
                n = torch.cat([nn, no], dim=0)
                p = all_positive_samples
                if len(p) < len(n):
                    p = torch.cat([p, over_sampler.generate_samples(len(n) - len(p)).cpu()])
                dataset = src.datasets.Dataset()
                dataset.samples = torch.cat([n, p], dim=0)
                dataset.labels = torch.cat(
                    [
                        torch.zeros(len(n)),
                        torch.ones(len(p))
                    ]
                )
                result.append(dataset)
        else:
            ns = all_negative_samples.split(math.ceil(len(all_negative_samples) / 3))
            ps = all_positive_samples.split(math.ceil(len(all_positive_samples) / 3))
            for n, p in zip(ns, ps):
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
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
        k = 5
        nn, no = [], []
        for i, sample in enumerate(neg_samples):
            indices = torch.topk(dists[i], k + 1, largest=False).indices[1:]
            labels = all_labels[indices]
            pos_num = int(sum(labels))
            if 1 <= pos_num <= 4:
                no.append(sample)
            elif pos_num == 0:
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

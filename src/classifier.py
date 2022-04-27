from math import sqrt

import torch
import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam, Optimizer
from sklearn.metrics import roc_auc_score, confusion_matrix

from src import config, logger, models
from src.types import Dataset, GAN


class Classifier:
    def __init__(self, name: str):
        self.name = str(name) + '_Classifier'
        self.model = models.ClassifierModel().to(config.device)
        self.logger = logger.Logger(name)
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
            'Precision': .0,
            'Recall': .0,
            'AACC': .0,
        }

    def fit(
            self,
            dataset: Dataset,
            gan: GAN = None,
    ):
        self.model.train()
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {config.device}')

        optimizer = Adam(
                params=self.model.parameters(),
                lr=config.classifier_config.lr,
                betas=(0.5, 0.9),
            )
        dataset.to(config.device)
        if gan is None:
            x, labels = dataset.samples, dataset.labels
        else:
            real_x, real_labels = dataset.samples, dataset.labels
            # get positive indices and negative indices
            pos_indices, neg_indices = [], []
            for idx, item in enumerate(real_labels):
                if item == 1:
                    pos_indices.append(idx)
                elif item == 0:
                    neg_indices.append(idx)
                else:
                    raise ValueError(f"Invalid value found in labels: {item}")
            # count positive samples and negative samples
            pos_num = len(pos_indices)
            neg_num = len(neg_indices)
            assert pos_num < neg_num
            generated_x_num = neg_num - pos_num
            generated_x = gan.generate_samples(generated_x_num)
            x = torch.cat([real_x, generated_x])
            labels = torch.cat([real_labels, torch.ones(len(generated_x), device=config.device)])

        for _ in range(config.classifier_config.epochs):
            self.model.zero_grad()
            prediction = self.model(x).squeeze()
            loss = binary_cross_entropy(
                input=prediction,
                target=labels,
            )
            loss.backward()
            optimizer.step()

        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor, use_prob: bool = False) -> torch.Tensor:
        x = x.to(config.device)
        prob = self.model(x).cpu()
        if use_prob:
            return prob.squeeze(dim=1).detach()
        else:
            return self._prob2label(prob).cpu()

    def test(self, test_dataset: Dataset):
        with torch.no_grad():
            x, label = test_dataset.samples.cpu(), test_dataset.labels.cpu()
            predicted_prob = self.predict(x, use_prob=True).cpu()
            predicted_label = self._prob2label(predicted_prob).cpu()
            tn, fp, fn, tp = confusion_matrix(
                y_true=label,
                y_pred=predicted_label,
            ).ravel()

            precision = tp / (tp + fp) if tp + fp != 0 else 0
            recall = tp / (tp + fn) if tp + fn != 0 else 0
            specificity = tn / (tn + fp) if tn + fp != 0 else 0

            tp_rate = recall
            tn_rate = specificity
            aacc = (tp_rate + tn_rate) / 2

            f1 = 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0
            g_mean = sqrt(recall * specificity)

            auc = roc_auc_score(
                y_true=label,
                y_score=predicted_prob,
            )

            self.metrics['F1'] = f1
            self.metrics['G-Mean'] = g_mean
            self.metrics['AUC'] = auc
            self.metrics['Precision'] = precision
            self.metrics['Recall'] = recall
            self.metrics['AACC'] = aacc

    @staticmethod
    def _prob2label(prob):
        probabilities = prob.squeeze()
        labels = np.zeros(probabilities.size())
        for i, p in enumerate(probabilities):
            if p >= 0.5:
                labels[i] = 1
        return torch.from_numpy(labels).to(config.device)

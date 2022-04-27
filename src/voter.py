import typing
from math import sqrt

import torch
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import src


class _SVC:
    def __init__(self):
        self.svc = SVC(random_state=src.config.seed)

    def fit(self, dataset: src.types.Dataset):
        x, y = dataset.samples, dataset.labels
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        self.svc.fit(x, y)

    def predict(self, x: torch.Tensor):
        x = x.numpy()
        return torch.tensor(
            self.svc.predict(x)
        )


class _DT:
    def __init__(self):
        self.dt = DecisionTreeClassifier(random_state=src.config.seed)

    def fit(self, dataset: src.types.Dataset):
        x, y = dataset.samples, dataset.labels
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        self.dt.fit(x, y)

    def predict(self, x: torch.Tensor):
        x = x.numpy()
        return torch.tensor(
            self.dt.predict(x)
        )


class Voter:
    def __init__(self):
        self.classifiers = [
            src.classifier.Classifier('V'),
            _SVC(),
            _DT(),
        ]
        self.logger = src.logger.Logger(self.__class__.__name__)
        self.metrics = {
            'F1': .0,
            'G-Mean': .0,
            'AUC': .0,
            'Precision': .0,
            'Recall': .0,
            'AACC': .0,
        }

    def fit(self, over_sampler: typing.Union[
        src.types.GAN,
        src.types.RGAN,
        SMOTE,
        ADASYN,
        SVMSMOTE,
    ] = None) -> None:
        self.logger.info('Started training')
        self.logger.debug(f'Using device: {src.config.device}')
        datasets = src.data_pipe.DataPipe().get_datasets(over_sampler)
        for classifier, dataset in zip(self.classifiers, datasets):
            classifier.fit(dataset)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(len(x))
        for classifier in self.classifiers:
            result += classifier.predict(x)
        for idx, item in enumerate(result):
            result[idx] = 1 if item >= 2 else 0
        return result

    def test(self, test_dataset: src.types.Dataset):
        with torch.no_grad():
            x, label = test_dataset.samples, test_dataset.labels
            predicted_label = self.predict(x)
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
                y_score=predicted_label,
            )

            self.metrics['F1'] = f1
            self.metrics['G-Mean'] = g_mean
            self.metrics['AUC'] = auc
            self.metrics['Precision'] = precision
            self.metrics['Recall'] = recall
            self.metrics['AACC'] = aacc

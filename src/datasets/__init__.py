from numpy import ndarray

from src.datasets._dataset import Dataset

from src.datasets.full_dataset import FullDataset
from src.datasets.negative_dataset import NegativeDataset
from src.datasets.positive_dataset import PositiveDataset
from src.datasets.weighted_positive_dataset import WeightedPositiveDataset


training_samples: ndarray = None
training_labels: ndarray = None
test_samples: ndarray = None
test_labels: ndarray = None

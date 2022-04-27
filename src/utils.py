import random

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

import src


def set_random_state(seed: int = None) -> None:
    if seed is None:
        seed = src.config.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def preprocess_data(file_name: str) -> (np.ndarray, np.ndarray):
    set_random_state()
    # concatenate the file path
    file_path = src.config.path_config.datasets / file_name
    # calculate skip rows
    skip_rows = 0
    occupied = True
    while occupied:
        try:
            with open(file_path, 'r') as f:
                while True:
                    line = f.readline()
                    if line[0] != '@':
                        break
                    else:
                        skip_rows += 1
            occupied = False
        except PermissionError:
            pass
    # read raw data
    df = pd.read_csv(file_path, sep=',', skiprows=skip_rows, header=None)
    np_array = df.to_numpy()
    np.random.shuffle(np_array)
    # partition labels and samples
    labels = np_array[:, -1].copy()
    samples = np_array[:, :-1].copy()
    # digitize labels
    for i, _ in enumerate(labels):
        labels[i] = labels[i].strip()
    labels[labels[:] == 'positive'] = 1
    labels[labels[:] == 'negative'] = 0
    labels = labels.astype('int')
    # normalize samples
    samples = minmax_scale(samples.astype('float32'))
    src.models.x_size = samples.shape[1]
    return samples, labels


def prepare_dataset(name: str, training_test_ratio: float = 0.8) -> None:
    samples, labels = preprocess_data(name)
    training_samples, test_samples, training_labels, test_labels = train_test_split(
        samples,
        labels,
        train_size=training_test_ratio,
        random_state=src.config.seed,
    )
    src.datasets.training_samples = training_samples
    src.datasets.training_labels = training_labels
    src.datasets.test_samples = test_samples
    src.datasets.test_labels = test_labels


def get_final_test_metrics(statistics: dict) -> dict:
    metrics = dict()
    for name, values in statistics.items():
        if name == 'Loss':
            continue
        else:
            metrics[name] = values[-1]
    return metrics


def get_balanced_dataset(imbalanced_dataset: src.types.Dataset, gan: src.types.GAN) -> src.types.Dataset:
    real_x, real_labels = imbalanced_dataset.samples, imbalanced_dataset.labels
    pos_num = int(sum(real_labels))
    neg_num = len(real_labels) - pos_num
    generated_x = gan.generate_samples(neg_num - pos_num)
    x = torch.cat([real_x, generated_x])
    labels = torch.cat([real_labels, torch.ones(len(generated_x), device=src.config.device)])
    balanced_dataset = src.datasets.Dataset()
    balanced_dataset.samples = x
    balanced_dataset.labels = labels
    balanced_dataset.to(src.config.device)
    return balanced_dataset


def turn_on_test_mode():
    print(f"""{"TEST MODE IS ON!":*^100}""")
    src.config.gan_config.epochs = 10
    src.config.classifier_config.epochs = 10



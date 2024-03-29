import torch

from . import (
    classifier_config,
    gan_config,
    logging_config,
    path_config,
    model_config,
)

# random seed
"""
WARNING: The random seed can only guarantee the reproducibility on the same computer with the same environment!
"""
seed = 1

# device used for training
device: str = 'auto'

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

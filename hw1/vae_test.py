"""Tests VAE example generation model."""

import sys
import torch
import torchvision
import data_provider
import itertools
import logging
from config import build_config
from models.vae_supervised import VAEExpander

logger = logging.getLogger('main')

# MNIST normalization parameters.
NORM_STD = 0.3081
NORM_MEAN = 0.1307
MIN = -NORM_MEAN / NORM_STD + 1e-4
MAX = (1-NORM_MEAN) / NORM_STD - 1e-4

# Which VAE model to use.
config = build_config(sys.argv[1], logger)

# Which example from data provider to modify.
example_id = int(sys.argv[2])

# How many samples to get.
n_trials = int(sys.argv[3])

vae = torch.load(config['vae_expander']['model_file_name'])
mode = config['vae_expander']['mode']

dp = data_provider.DataProvider('train_labeled.p')

vaes = VAEExpander(config, vae)
vaes.build(dp.loader)

for i in range(n_trials):
    examples = list(itertools.islice(vaes, example_id + 1))
    z = (examples[example_id][0] * NORM_STD + NORM_MEAN).clamp(0, 1)
    file_name = '{}/vae_test_{}_{:02}_{:02}.png'.format(mode, mode, example_id, i)
    torchvision.utils.save_image(z, file_name, nrow=8)
    print(file_name)

vaes.print_info()

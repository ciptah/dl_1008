"""Tests VAE example generation model."""

import sys
import torch
import torchvision
import data_provider
import itertools
from models.vae_supervised import VAEExpander

# MNIST normalization parameters.
NORM_STD = 0.3081
NORM_MEAN = 0.1307
MIN = -NORM_MEAN / NORM_STD + 1e-4
MAX = (1-NORM_MEAN) / NORM_STD - 1e-4

# Which VAE model to use.
model = sys.argv[1]

# Which example from data provider to modify.
example_id = int(sys.argv[2])

# How many samples to get.
n_trials = 10

mode = 'hull'

vae = torch.load(model)

dp = data_provider.DataProvider('train_labeled.p')

vaes = models.vae_supervised.VAEExpander({}, vae)
vaes.build(dp.loader)

for i in range(n_trials):
    examples = list(itertools.islice(vaes.hull_mode(), example_id + 1))
    z = (examples[example_id][0] * NORM_STD + NORM_MEAN).clamp(0, 1)
    file_name = '{}_test_{}_{:02}.png'.format(sys.argv[1], mode, i)
    torchvision.utils.save_image(z, file_name, nrow=8)
    print(file_name)

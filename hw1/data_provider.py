"""
Module that contains all code to pre-process and provide data for a predicitve model
"""
import torch
import pickle
import logging
from torchvision import datasets, transforms
import numpy as np

logger = logging.getLogger('data_provider')

class Loader(object):
    """Base data loader wrapper class.
    
    We do a lot of input duplication and manipulation so this class is used
    to make the counts correct."""
    def __init__(self, torch_loader):
        super(Loader, self).__init__()
        self.torch_loader = torch_loader

    def __iter__(self):
        return iter(self.torch_loader)

    def __len__(self):
        """Number of minibatches."""
        return len(self.torch_loader)

    def example_count(self):
        """Number of training examples."""
        return len(self.torch_loader.dataset)

class DataProvider:
    """
    Class that serves data to a model
    """
    def __init__(self, file_dir=None, train=False, batch_size=64):
        self.train = train

        # load dataset
        if file_dir is None:
            logger.info('Downloading dataset using torchvision')
            self.dataset = datasets.MNIST('.', download=True, train=self.train, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        else:
            logger.info('opening pickle file at %s', file_dir)
            with open(file_dir, "rb") as f:
                self.dataset = pickle.load(f)

        # create loader
        self.loader = Loader(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=self.train))

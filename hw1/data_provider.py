"""
Module that contains all code to pre-process and provide data for a predicitve model
"""
import torch
import pickle
from torchvision import datasets, transforms
import numpy as np


class DataProvider:
    """
    Class that serves data to a model
    """
    def __init__(self, file_dir=None, train=False, batch_size=64):
        self.train = train

        # load dataset
        if file_dir is None:
            self.dataset = datasets.MNIST('.', download=True, train=self.train, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        else:
            with open(file_dir, "rb") as f:
                self.dataset = pickle.load(f)

        # create loader
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=self.train)

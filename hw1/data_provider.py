"""
Module that contains all code to pre-process and provide data for a predicitve model
"""
import torch
import pickle
import logging
from torchvision import datasets, transforms

logger = logging.getLogger('DataProvider')

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
        self.loader = torch.utils.data.DataLoader(self.dataset,batch_size=batch_size, shuffle=self.train)


class UnlabeledProvider:
    def __init__(self, file_dir='train_unlabeled.p', shuffle=True, batch_size=64):
        self.shuffle = shuffle

        logger.info('opening pickle file at %s', file_dir)
        with open(file_dir, "rb") as f:
            self.dataset = pickle.load(f)
        self.dataset.train_labels = torch.zeros(57000)

        # create loader
        self.loader = torch.utils.data.DataLoader(self.dataset,
                batch_size=batch_size, shuffle=self.shuffle)

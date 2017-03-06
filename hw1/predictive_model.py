"""
Module that contains all code for the MNIST predictive model
"""

import torch.optim as optim
import constants
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


logger = logging.getLogger('predictive_model')

class PredictiveModel:
    """
    Class that represents a model
    """
    def __init__(self, config):
        learning_rate = config['training']['learning_rate']
        momentum = config['training']['momentum']['mu_coefficient']
        self.config = config
        model_name = config.get('model', 'basic')
        logger.info('Using model "%s".', model_name)
        self.model = constants.models[model_name](config)
        # TODO: optimization method should probably be configurable too.
        self.optimizer = optim.SGD(self.model.parameters(),
                lr=learning_rate, momentum=momentum)

    def start_train(self):
        self.model.train()

    def start_prediction(self):
        self.model.eval()

    def train_batch(self, data, target, **kwargs):
        """
        Method that trains a batch of data
        :param data:
        :param target:
        :param kwargs:
        :return:
        """
        # data: 64 * 1 * 28 * 28
        # target: 64
        self.optimizer.zero_grad()
        output = self.model(data)
        # output: 64 * 10
        loss = self.model.loss_func(output, target)
        # loss:  (1,)
        loss.backward()
        self.optimizer.step()
        return output, loss.data[0]

    def validate_batch(self, data, target):
        """
        Method that validates a batch of data
        :param data:
        :param target:
        :return:
        """
        output, pred = self.predict_batch(data)
        loss_val = self.model.loss_func(output, target).data[0]
        return output, pred, loss_val

    def predict_batch(self, data):
        """
        Method that predicts a batch of data
        :param data:
        :return:
        """
        output = self.model(data)
        pred = output[0].data.max(1)[1]  # get the index of the max log-probability
        return output, pred

def get_unlabeled_model(config):
    model = config.get('unlabeled_model', None)
    logger.info('Unlabeled model "%s".', model)
    if not model:
        return None
    return constants.unlabeled_models[model](config)

def augment_training_data(config, loader):
    setting = config.get('augment_training_data', 'default')
    if setting:
        loader = constants.augment_training[setting](config, loader)
    return loader

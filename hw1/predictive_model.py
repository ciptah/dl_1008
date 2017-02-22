"""
Module that contains all code for the MNIST predictive model
"""
import torch.nn.functional as F
import torch.optim as optim

import constants
import logging

logger = logging.getLogger('predictive_model')

class PredictiveModel:
    """
    Class that represents a model
    """
    def __init__(self, config):
        learning_rate = config['training']['learning_rate']
        momentum = config['training']['momentum']['mu_coefficient']

        self.config = config

        model = config.get('model', 'basic')
        logger.info('Using model "%s".', model)
        self.model = constants.models[model](config)

        # TODO: optimization method should probably be configurable too.
        self.optimizer = optim.SGD(self.model.parameters(),
                lr=learning_rate, momentum=momentum)
        self.loss_func = F.nll_loss

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
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_func(output, target)
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
        loss_val = self.loss_func(output, target).data[0]
        return output, pred, loss_val

    def predict_batch(self, data):
        """
        Method that predicts a batch of data
        :param data:
        :return:
        """
        output = self.model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        return output, pred




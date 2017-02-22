"""
Module that contains all code for the MNIST predictive model
"""
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import logging

logger = logging.getLogger('predictive_model')

class BasicNet(nn.Module):
    """
    Sample Model.

    This is the model from the example code, adapted a bit so the momentum
    and dropout parameters can be tuned via configuration parameters.
    """
    def __init__(self, dropout=0.5):
        super(Net, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=self.dropout)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Input is 1x28x28.
        # N (minibatch size) is implicit.

        # Conv1: 1x28x28 -> 10x24x24
        # Pool: 10x24x24 -> 10x12x12
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv2: 10x12x12 -> 20x8x8
        # Pool: 20x8x8 -> 20x4x4
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # View: 20x4x4 -> 320
        x = x.view(-1, 320)
        # FC: 320 -> 50
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        # FC: 50->10
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class PredictiveModel:
    """
    Class that represents a model
    """
    def __init__(self, config):
        learning_rate = config['training']['learning_rate']
        dropout = config['training']['dropout']
        momentum = config['training']['momentum']['mu_coefficient']

        self.config = config
        self.model = Net(dropout=dropout)

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




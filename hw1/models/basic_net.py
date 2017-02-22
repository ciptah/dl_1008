"""
Module that contains the basic neural net model.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import logging

logger = logging.getLogger('basic_net')

class Net(nn.Module):
    """
    Sample Model.

    This is the model from the example code, adapted a bit so the momentum
    and dropout parameters can be tuned via configuration parameters.

    It's a ConvNet with 2 convolution and 2 fully connected layers.

    Worth noting that the nn.Conv2d and nn.Linear modules provided by PyTorch
    use the scaled initialization standard deviation found in He et al.
    """
    def __init__(self, dropout=0.5, prelu=False):
        super(Net, self).__init__()
        self.dropout = dropout

        # TODO: Configurable sizes/channels/parameters.
        conv1_channels = 10
        conv2_channels = 20

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=self.dropout)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.prelu = prelu
        if prelu:
            logger.info('using PReLU.')
            self.conv1_prelu = nn.PReLU(10)
            self.conv2_prelu = nn.PReLU(20)
            self.fc1_prelu = nn.PReLU(50)
            self.fc2_prelu = nn.PReLU(10)
        else:
            self.register_parameter('conv1_prelu', None)
            self.register_parameter('conv2_prelu', None)
            self.register_parameter('fc1_prelu', None)
            self.register_parameter('fc2_prelu', None)

        logger.info('dropout = %f', dropout)
        logger.info('%d channels for 1st convnet', conv1_channels)
        logger.info('%d channels for 2nd convnet', conv2_channels)

    def relu(self, input, which_prelu):
        """Pick which rectifier unit to use, ReLU or PReLU."""
        if self.prelu:
            return which_prelu(input)
        else:
            return F.relu(input)

    def forward(self, x):
        # Input is 1x28x28.
        # N (minibatch size) is implicit.

        # Conv1: 1x28x28 -> 10x24x24
        # Pool: 10x24x24 -> 10x12x12
        x = self.relu(F.max_pool2d(self.conv1(x), 2), self.conv1_prelu)
        # Conv2: 10x12x12 -> 20x8x8
        # Pool: 20x8x8 -> 20x4x4
        x = self.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2), self.conv2_prelu)
        # View: 20x4x4 -> 320
        x = x.view(-1, 320)
        # FC: 320 -> 50
        x = self.relu(self.fc1(x), self.fc1_prelu)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # FC: 50->10
        x = self.relu(self.fc2(x), self.fc2_prelu)
        return F.log_softmax(x)

def basicNet(config):
    dropout = config.get('training', {}).get('dropout', '0.5')
    prelu = config.get('training', {}).get('PReLU', False)
    return Net(dropout=dropout, prelu=prelu)

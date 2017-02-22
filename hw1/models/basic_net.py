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
    """
    def __init__(self, dropout=0.5):
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

        logger.info('dropout = %f', dropout)
        logger.info('%d channels for 1st convnet', conv1_channels)
        logger.info('%d channels for 2nd convnet', conv2_channels)

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

def basicNet(config):
    dropout = config['training']['dropout']
    return Net(dropout=dropout)

"""
Module that contains the basic neural net model.
"""
import torch.nn.functional as F
import torch.nn as nn
import constants
import logging
from torch.autograd import Variable

logger = logging.getLogger('improved_net')


class ImprovedNet(nn.Module):
    """
    Sample Model
    """
    def __init__(self, psuedo_label_alpha_func=None):
        super(ImprovedNet, self).__init__()
        self.dropout = 0.4

        # The head of the network changes at epoch 30.
        self.conv1a = nn.Conv2d(1, 60, padding=2, kernel_size=5) # 28 -> 28
        self.conv1b = nn.Conv2d(60, 60, kernel_size=5) # 28 -> 24 -> 12

        self.conv1 = nn.Conv2d(1, 60, kernel_size=5) # 28 -> 24 -> 12

        self.conv2 = nn.Conv2d(60, 90, kernel_size=3)  # 12 -> 10
        self.conv3 = nn.Conv2d(90, 100, kernel_size=3) # 10 -> 8 -> 4
        self.fc1 = nn.Linear(1600, 800)
        self.fc1a = nn.Linear(800, 800)
        self.fc2 = nn.Linear(800, 10)
        self.psuedo_label_alpha_func = psuedo_label_alpha_func
        self.current_epoch_num = 1

    def forward(self, x):
        if self.current_epoch_num > 30:
            x = self.conv1a(x)
        else:
            x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # after relu: 64*10*12*12

        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv2(x))
        # after second conv pool 64 * 20 * 10 * 10

        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # after third conv pool 64 * 20 * 4 * 4

        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=self.dropout)
        if self.current_epoch_num > 10:
            x = F.relu(self.fc1a(x))
        x = self.fc2(x)
        return [F.log_softmax(x)]

    def loss_func(self, output, target):
        unlabel_epoch = (-1 in target.data)
        if unlabel_epoch:
            if self.psuedo_label_alpha_func is not None:
                max_class = output[0].data.max(1)[1]
                max_class = max_class.view(-1)
                target = Variable(max_class)
            else:
                raise ValueError('pseudo_label_alpha_func must not be None')
            loss_nll = F.nll_loss(output[0], target)
        else:
            loss_nll = F.nll_loss(output[0], target)
        if self.psuedo_label_alpha_func is not None:
            loss_nll *= self.psuedo_label_alpha_func(unlabel_epoch, self.current_epoch_num)
        return loss_nll


def improvedNet(config):
    pseudo_func_key = config.get('pseudo_label_func', 'default')
    pseudo_func = None if pseudo_func_key is None else constants.psuedo_label_func_dict[pseudo_func_key]
    return ImprovedNet(psuedo_label_alpha_func=pseudo_func)

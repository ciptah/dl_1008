"""
Module that contains the basic neural net model.
"""
import torch.nn.functional as F
import torch.nn as nn
import constants
import logging
from torch.autograd import Variable

logger = logging.getLogger('basic_net')


class Net(nn.Module):
    """
    Sample Model
    """
    def __init__(self, psuedo_label_alpha_func=None):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.psuedo_label_alpha_func = psuedo_label_alpha_func
        self.current_epoch_num = 1

    def forward(self, x):
        # x: 64 * 1 * 28 * 28
        x = self.conv1(x)
        # after conv: 64*10*24*24
        x = F.max_pool2d(x, 2)
        # after pool 64 * 10 * 12 * 12
        x = F.relu(x)
        # after relu: 64*10*12*12
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
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


def basicNet(config):
    pseudo_func_key = config.get('pseudo_label_func', 'default')
    pseudo_func = None if pseudo_func_key is None else constants.psuedo_label_func_dict[pseudo_func_key]
    return Net(psuedo_label_alpha_func=pseudo_func)
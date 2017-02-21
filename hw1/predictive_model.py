"""
Module that contains all code for the predictive model
"""
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    """
    Sample Model
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


class PredictiveModel:
    """
    Class that represents a model
    """
    def __init__(self, config):
        self.config = config
        self.model = Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.05)
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




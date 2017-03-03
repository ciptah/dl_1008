"""
Module that contains all code for the predictive model
"""
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



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
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

    def loss_func(self, output, target):
        return [F.nll_loss(output, target)]


class SWWAE(nn.Module):
    """
    Stacked What-Where Auto-Encoders
    """
    def __init__(self, lambda_rec=0.41, lambda_m=0.41):
        super(SWWAE, self).__init__()
        # init var
        self.lambda_rec = lambda_rec
        self.lambda_m = lambda_m
        self.conv1_kernel_size = 10
        self.conv2_kernel_size = 20

        # encoder
        self.encoder_conv1 = nn.Conv2d(1, self.conv1_kernel_size, kernel_size=5)
        self.encoder_conv2 = nn.Conv2d(self.conv1_kernel_size, self.conv2_kernel_size, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.encoder_fc1 = nn.Linear(self.conv2_kernel_size*4*4, 50)
        self.encoder_fc2 = nn.Linear(50, 10)
        # decoder
        self.decoder_conv1 = nn.ConvTranspose2d(self.conv1_kernel_size, 1, kernel_size=5)
        self.decoder_conv2 = nn.ConvTranspose2d(self.conv2_kernel_size, self.conv1_kernel_size, kernel_size=5)
        self.decoder_unpool2 = nn.MaxUnpool2d(2)
        self.decoder_unpool1 = nn.MaxUnpool2d(2)

    def forward(self, x):
        # encoder: x = 28 * 28
        x_val = F.relu(self.encoder_conv1(x)) # 24 * 24
        x_encode_m1, encode_where_1 = F.max_pool2d(x_val, 2, return_indices=True) # 12 * 12
        x_val = F.relu(self.encoder_conv2(x_encode_m1)) # 8 * 8
        x_val = self.conv2_drop(x_val)
        x_encode_m2, encode_where_2 = F.max_pool2d(x_val, 2, return_indices=True) # 4 * 4
        # prediction
        x_out = x_encode_m2.view(-1, self.conv2_kernel_size*4*4)
        x_out = F.relu(self.encoder_fc1(x_out))
        x_out = F.dropout(x_out, training=self.training)
        x_out = F.relu(self.encoder_fc2(x_out))
        x_out = F.log_softmax(x_out)

        # decoder
        x_val = self.decoder_unpool2(x_encode_m2, encode_where_2) # 8 * 8
        x_decode_m1 = F.relu(self.decoder_conv2(x_val)) # 12 * 12
        x_val = self.decoder_unpool1(x_decode_m1, encode_where_1) # 24 * 24
        x_rec = self.decoder_conv1(x_val) # 28 * 28
        return x_out, x_encode_m1, x_decode_m1, x_rec, x


    def loss_func(self, output, target):
        x_out, x_encode_m1, x_decode_m1, x_rec, x = output
        n = x.size()[0]
        # unsupervised samples should have -1 label
        # pseudo-label
        # if -1 in target.data:
        #     max_class = x_out.data.max(1)[1]
        #     max_class = max_class.view(-1)
        #     target = Variable(max_class)
        # loss_nll = F.nll_loss(x_out, target)
        loss_nll = F.nll_loss(x_out, target) if -1 not in target.data else 0
        # L2m
        sq_diff_m1 = (x_encode_m1-x_decode_m1)**2
        sq_diff_m1 = sq_diff_m1.view(n,-1)
        m1_dim = sq_diff_m1.size()[1]
        loss_m1 = sq_diff_m1.sum(1).sum() / n / m1_dim
        # L2rec
        sq_diff_rec = ((x-x_rec)**2)
        sq_diff_rec = sq_diff_rec.view(n,-1)
        rec_dim = sq_diff_rec.size()[1]
        loss_rec = sq_diff_rec.sum(1).sum() / n / rec_dim
        return loss_nll + self.lambda_m * loss_m1 + self.lambda_rec * loss_rec


class EnsembleModel:
    """
    Class that represents a ensemble of predictive model
    """
    def __int__(self, models):
        self.models = models

    def start_train(self):
        for model in self.models:
            model.train()

    def start_prediction(self):
        for model in self.models:
            model.eval()

    def validate_batch(self, data, target):
        """
        Method that validates a batch of data
        :param data:
        :param target:
        :return:
        """
        #for model in self.models:
        #    output, _ = model.predict_batch(data)
        #output = output.mean()

        #return output, pred, None

    def predict_batch(self, data):
        """
        Method that predicts a batch of data
        :param data:
        :return:
        """
        output_list = []
        for model in self.models:
            output = model(data)
            output_list.append(output)
        pred = output[0].data.max(1)[1]  # get the index of the max log-probability
        return output, pred


class PredictiveModel:
    """
    Class that represents a model
    """
    def __init__(self, config):
        self.config = config
        #self.model = Net()
        self.model = SWWAE()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.05)

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




"""
The SWWAE Model
"""

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import constants


class SWWAE(nn.Module):
    """
    Stacked What-Where Auto-Encoders
    """
    def __init__(self, lambda_rec=0.41, lambda_m=0.41, psuedo_label_alpha_func=None):
        super(SWWAE, self).__init__()
        # algorithm specific variables
        self.psuedo_label_alpha_func = psuedo_label_alpha_func
        self.current_epoch_num = 0

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
        #x_out = F.dropout(x_out, training=self.training)
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
        unlabel_epoch = (-1 in target.data)
        # pseudo-label nll loss
        if self.psuedo_label_alpha_func is not None:
            if unlabel_epoch:
                max_class = x_out.data.max(1)[1]
                max_class = max_class.view(-1)
                target = Variable(max_class)
            loss_nll = F.nll_loss(x_out, target)
        # normal nll loss
        else:
            loss_nll = 0 if unlabel_epoch else F.nll_loss(x_out, target)
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
        # total loss
        total_loss = loss_nll + self.lambda_m * loss_m1 + self.lambda_rec * loss_rec
        if self.psuedo_label_alpha_func is not None:
            total_loss *= self.psuedo_label_alpha_func(unlabel_epoch, self.current_epoch_num)
        return total_loss

def swwae(config):
    pseudo_func_key = config.get('pseudo_label_func', None)
    pseudo_func = None if pseudo_func_key is None else constants.psuedo_label_func_dict[pseudo_func_key]
    return SWWAE(psuedo_label_alpha_func=pseudo_func)
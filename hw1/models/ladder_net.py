"""
Module that contains the ladder neural net model.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch

import logging
from models import basic_net
from torch.autograd import Variable

logger = logging.getLogger('ladder_net')

class CMul(nn.Module):
    """
    This net implements element-wise add and multiplication 
    """
    def __init__(self, tensor_size):
        super(CMul, self).__init__()
        self.beta = nn.Parameter(torch.randn(1, tensor_size))
        self.gamma = nn.Parameter(torch.randn(1, tensor_size))
        
    def forward(self, x):
        return (x + self.beta.expand(x.size())) * self.gamma.expand(x.size())

class Decoder(nn.Module):
    """
    This net implements decoding part
    """
    def __init__(self, tensor_size):
        super(Decoder, self).__init__()
        init_weights = 1e-2
        self.a1 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a2 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a3 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a4 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a5 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a6 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a7 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a8 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a9 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.a10 = nn.Parameter(torch.randn(1, tensor_size) * init_weights)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z, u):
        
        mu = self.a1.expand(u.size())*self.sigmoid(self.a2.expand(u.size())*u+self.a3.expand(u.size()))+self.a4.expand(u.size())*u+self.a5.expand(u.size())
        v = self.a6.expand(u.size())*self.sigmoid(self.a7.expand(u.size())*u+self.a8.expand(u.size()))+self.a9.expand(u.size())*u+self.a10.expand(u.size())
        return (z - mu) * v + mu               
        

class LadderNet(basic_net.Net):
    """
    This model implements the ladder neural network proposed in Semi-Supervised 
    Learning with Ladder Networks by A Rasmus, H Valpola, M Honkala, 
    M Berglund, and T Raiko
    """
    def __init__(self, dropout=0.5, prelu=False, noise=0.1):
        super(LadderNet, self).__init__()
        self.dropout = dropout
        self.noise = noise
        self.layers = [1000, 500, 250, 250, 250, 10]
        self.enc1 = nn.Linear(784, self.layers[0])
        self.n1 = nn.BatchNorm1d(self.layers[0])
        self.h1 = CMul(self.layers[0])
        self.enc2 = nn.Linear(self.layers[0], self.layers[1])
        self.n2 = nn.BatchNorm1d(self.layers[1])
        self.h2 = CMul(self.layers[1])
        self.enc3 = nn.Linear(self.layers[1], self.layers[2])
        self.n3 = nn.BatchNorm1d(self.layers[2])
        self.h3 = CMul(self.layers[2])
        self.enc4 = nn.Linear(self.layers[2], self.layers[3])
        self.n4 = nn.BatchNorm1d(self.layers[3])
        self.h4 = CMul(self.layers[3])
        self.enc5 = nn.Linear(self.layers[3], self.layers[4])
        self.n5 = nn.BatchNorm1d(self.layers[4])
        self.h5 = CMul(self.layers[4])
        #Top layer
        self.enc6 = nn.Linear(self.layers[4], self.layers[5])
        self.n6 = nn.BatchNorm1d(self.layers[5])
        self.h6 = CMul(self.layers[5])        
        self.dn6 = nn.BatchNorm1d(self.layers[5])
        self.dec6 = Decoder(self.layers[5])
        self.v6 = nn.Linear(self.layers[5], self.layers[4])
        self.dn5 = nn.BatchNorm1d(self.layers[4])
        self.dec5 = Decoder(self.layers[4])
        self.v5 = nn.Linear(self.layers[4], self.layers[3])
        self.dn4 = nn.BatchNorm1d(self.layers[3])
        self.dec4 = Decoder(self.layers[3])
        self.v4 = nn.Linear(self.layers[3], self.layers[2])
        self.dn3 = nn.BatchNorm1d(self.layers[2])
        self.dec3 = Decoder(self.layers[2])
        self.v3 = nn.Linear(self.layers[2], self.layers[1])
        self.dn2 = nn.BatchNorm1d(self.layers[1])
        self.dec2 = Decoder(self.layers[1])
        self.v2 = nn.Linear(self.layers[1], self.layers[0])
        self.dn1 = nn.BatchNorm1d(self.layers[0])
        self.dec1 = Decoder(self.layers[0])
        self.v1 = nn.Linear(self.layers[0], 784)
        self.dn0 = nn.BatchNorm1d(784)
        self.dec0 = Decoder(784)

        self.prelu = prelu
        if prelu:
            logger.info('using PReLU.')
            #TODO
        else:
            #TODO
            self.register_parameter('conv1_prelu', None)

            
    def addNoise(self, x, noise):
        sigma = float(noise)**2
        N = x.size()
        #return torch.normal(torch.zeros(N), torch.ones(N)*sigma) + x
        b = torch.randn(*x.size())
        #b = torch.normal(torch.zeros(N), torch.ones(N)*sigma)
        x = x + torch.autograd.Variable(b * sigma,volatile=False)
        return x

    def relu(self, input, which_prelu):
        """Pick which rectifier unit to use, ReLU or PReLU."""
        if self.prelu:
            return which_prelu(input)
        else:
            return F.relu(input)

    def forward(self, x):
        #clean path
        x = x.view(-1, 784)
        z_clean_0 = x
        h_clean_0 = z_clean_0
        z_pre_1 = self.enc1(h_clean_0)
        z_clean_1 = self.n1(z_pre_1)
        h_clean_1 = self.relu(self.h1(z_clean_1), self.prelu)
        z_pre_2 = self.enc2(h_clean_1)
        z_clean_2 = self.n2(z_pre_2)
        h_clean_2 = self.relu(self.h2(z_clean_2), self.prelu)
        z_pre_3 = self.enc3(h_clean_2)
        z_clean_3 = self.n3(z_pre_3)
        h_clean_3 = self.relu(self.h3(z_clean_3), self.prelu)
        z_pre_4 = self.enc4(h_clean_3)
        z_clean_4 = self.n4(z_pre_4)
        h_clean_4 = self.relu(self.h4(z_clean_4), self.prelu)
        z_pre_5 = self.enc5(h_clean_4) 
        z_clean_5 = self.n5(z_pre_5)
        h_clean_5 = self.relu(self.h5(z_clean_5), self.prelu)
        z_pre_6 = self.enc6(h_clean_5)
        z_clean_6 = self.n6(z_pre_6)
        h_clean_6 = self.relu(self.h6(z_clean_6), self.prelu)
        
        #corrupted path
        z_corrupt_0 = self.addNoise(x, self.noise)
        h_corrupt_0 = z_corrupt_0
        z_corrupt_1 = self.addNoise(self.n1(self.enc1(h_corrupt_0)), self.noise)
        h_corrupt_1 = self.relu(self.h1(z_corrupt_1), self.prelu)
        z_corrupt_2 = self.addNoise(self.n2(self.enc2(h_corrupt_1)), self.noise)
        h_corrupt_2 = self.relu(self.h2(z_corrupt_2), self.prelu)
        z_corrupt_3 = self.addNoise(self.n3(self.enc3(h_corrupt_2)), self.noise)
        h_corrupt_3 = self.relu(self.h3(z_corrupt_3), self.prelu)
        z_corrupt_4 = self.addNoise(self.n4(self.enc4(h_corrupt_3)), self.noise)
        h_corrupt_4 = self.relu(self.h4(z_corrupt_4), self.prelu)
        z_corrupt_5 = self.addNoise(self.n5(self.enc5(h_corrupt_4)), self.noise)
        h_corrupt_5 = self.relu(self.h5(z_corrupt_5), self.prelu)
        z_corrupt_6 = self.addNoise(self.n6(self.enc6(h_corrupt_5)), self.noise)
        h_corrupt_6 = self.relu(self.h6(z_corrupt_6), self.prelu)
        
        #decode
        u_6 = self.dn6(h_corrupt_6)
        z_hat_6 = self.dec6(z_corrupt_6, u_6)
        z_decode_6 = (z_hat_6 - torch.mean(z_pre_6, 0).expand(z_pre_6.size())) / Variable(torch.std(z_pre_6.data, 0).expand(z_pre_6.size()), requires_grad=False)
        #z_decode_6.data = z_decode_6.data / torch.std(z_pre_6.data, 0).expand(z_pre_6.size())
        u_5 = self.dn5(self.v6(z_hat_6))
        z_hat_5 = self.dec5(z_corrupt_5, u_5)
        z_decode_5 = (z_hat_5 - torch.mean(z_pre_5, 0).expand(z_pre_5.size())) / Variable(torch.std(z_pre_5.data, 0).expand(z_pre_5.size()), requires_grad=False)
        #z_decode_5.data = z_decode_5.data / torch.std(z_pre_5.data, 0).expand(z_pre_5.size())
        u_4 = self.dn4(self.v5(z_hat_5))
        z_hat_4 = self.dec4(z_corrupt_4, u_4)
        z_decode_4 = (z_hat_4 - torch.mean(z_pre_4, 0).expand(z_pre_4.size())) / Variable(torch.std(z_pre_4.data, 0).expand(z_pre_4.size()), requires_grad=False)
        #z_decode_4.data = z_decode_4.data / torch.std(z_pre_4.data, 0).expand(z_pre_4.size())
        u_3 = self.dn3(self.v4(z_hat_4))
        z_hat_3 = self.dec3(z_corrupt_3, u_3)
        z_decode_3 = (z_hat_3 - torch.mean(z_pre_3, 0).expand(z_pre_3.size())) / Variable(torch.std(z_pre_3.data, 0).expand(z_pre_3.size()), requires_grad=False)
        #z_decode_3.data = z_decode_3.data / torch.std(z_pre_3.data, 0).expand(z_pre_3.size())
        u_2 = self.dn2(self.v3(z_hat_3))
        z_hat_2 = self.dec2(z_corrupt_2, u_2)
        z_decode_2 = (z_hat_2 - torch.mean(z_pre_2, 0).expand(z_pre_2.size())) / Variable(torch.std(z_pre_2.data, 0).expand(z_pre_2.size()), requires_grad=False)
        #z_decode_2.data = z_decode_2.data / torch.std(z_pre_2.data, 0).expand(z_pre_2.size())
        u_1 = self.dn1(self.v2(z_hat_2))
        z_hat_1 = self.dec1(z_corrupt_1, u_1)
        z_decode_1 = (z_hat_1 - torch.mean(z_pre_1, 0).expand(z_pre_1.size())) / Variable(torch.std(z_pre_1.data, 0).expand(z_pre_1.size()), requires_grad=False)
        #z_decode_1.data = z_decode_1.data / torch.std(z_pre_1.data, 0).expand(z_pre_1.size())
        u_0 = self.dn0(self.v1(z_hat_1))
        z_hat_0 = self.dec0(z_corrupt_0, u_0)
        z_decode_0 = z_hat_0        
    
        #calculate distance between z_decode_i and z_clean_i
        d0 = ((z_decode_0 - z_clean_0)**2).mean()
        d1 = ((z_decode_1 - z_clean_1)**2).mean()
        d2 = ((z_decode_2 - z_clean_2)**2).mean()
        d3 = ((z_decode_3 - z_clean_3)**2).mean()
        d4 = ((z_decode_4 - z_clean_4)**2).mean()
        d5 = ((z_decode_5 - z_clean_5)**2).mean()
        d6 = ((z_decode_6 - z_clean_6)**2).mean()
        y_corrupt = F.log_softmax(h_corrupt_6)
        y_clean = F.log_softmax(h_clean_6)
        return y_clean, 1000*d0+1*d1+0.01*(d2+d3+d4+d5+d6), y_corrupt
        #return y, (d1+d2+d3+d4+d5+d6+d0)*10**-10
    
    def loss_func(self, output, target):
        y_clean, recon_error, y_hat = output
        # unsupervised samples should have -1 label
        # pseudo-label
        # if -1 in target.data:
        #     max_class = x_out.data.max(1)[1]
        #     max_class = max_class.view(-1)
        #     target = Variable(max_class)
        # loss_nll = F.nll_loss(x_out, target)
        loss_nll = F.nll_loss(y_hat, target) #if -1 not in target.data else 0
        # Reconstruction Error
        return loss_nll + recon_error

def ladderNet(config):
    dropout = config.get('training', {}).get('dropout', '0.5')
    prelu = config.get('training', {}).get('PReLU', False)
    noise = config.get('training', {}).get('noise', '0.3')
    return LadderNet(dropout=dropout, prelu=prelu, noise=noise)

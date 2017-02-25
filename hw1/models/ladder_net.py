"""
Module that contains the ladder neural net model.
"""
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

import logging
from models import basic_net
from nltk.sem.logic import Variable

logger = logging.getLogger('ladder_net')

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
        self.layers = [500, 100, 50, 20]
        self.enc1 = nn.Linear(784, self.layers[0])
        self.n1 = nn.BatchNorm1d(self.layers[0])
        self.enc2 = nn.Linear(self.layers[0], self.layers[1])
        self.n2 = nn.BatchNorm1d(self.layers[1])
        self.enc3 = nn.Linear(self.layers[1], self.layers[2])
        self.n3 = nn.BatchNorm1d(self.layers[2])
        self.enc4 = nn.Linear(self.layers[2], self.layers[3])
        self.n4 = nn.BatchNorm1d(self.layers[3])
        self.enc5 = nn.Linear(self.layers[3], 10)
        self.n5 = nn.BatchNorm1d(10)
        
        self.dec4 = nn.Linear(self.layers[3]+10, self.layers[3])
        self.dn4 = nn.BatchNorm1d(self.layers[3])
        self.dec3 = nn.Linear(self.layers[2]+self.layers[3], self.layers[2])
        self.dn3 = nn.BatchNorm1d(self.layers[2])
        self.dec2 = nn.Linear(self.layers[1]+self.layers[2], self.layers[1])
        self.dn2 = nn.BatchNorm1d(self.layers[1])
        self.dec1 = nn.Linear(self.layers[0]+self.layers[1], self.layers[0])
        self.dn1 = nn.BatchNorm1d(self.layers[0])
        
        self.prelu = prelu
        if prelu:
            logger.info('using PReLU.')
            #TODO
        else:
            #TODO
            self.register_parameter('conv1_prelu', None)

            
    def addNoise(self, x, noise):
        sigma = float(noise)
        N = x.size()
        #return torch.normal(torch.zeros(N), torch.ones(N)*sigma) + x
        b = nn.Parameter(torch.randn(*x.size()))
        #b = torch.normal(torch.zeros(N), torch.ones(N)*sigma)
        return x + b*sigma

    def relu(self, input, which_prelu):
        """Pick which rectifier unit to use, ReLU or PReLU."""
        if self.prelu:
            return which_prelu(input)
        else:
            return F.relu(input)
        
    def forward(self, x):
        #clean path
        x = x.view(-1, 784)
        x_clean_1 = self.relu(self.n1(self.enc1(x)), self.prelu)
        x_clean_2 = self.relu(self.n2(self.enc2(x_clean_1)), self.prelu)
        x_clean_3 = self.relu(self.n3(self.enc3(x_clean_2)), self.prelu)
        x_clean_4 = self.relu(self.n4(self.enc4(x_clean_3)), self.prelu)
        x_clean_5 = self.relu(self.n5(self.enc5(x_clean_4)), self.prelu)
        #corrupted path
        x_corrupt_1 = self.n1(self.enc1(x))
        x_corrupt_1 = self.relu(self.addNoise(x_corrupt_1, self.noise), self.prelu)
        x_corrupt_2 = self.n2(self.enc2(x_corrupt_1))
        x_corrupt_2 = self.relu(self.addNoise(x_corrupt_2, self.noise), self.prelu)
        x_corrupt_3 = self.n3(self.enc3(x_corrupt_2))
        x_corrupt_3 = self.relu(self.addNoise(x_corrupt_3, self.noise), self.prelu)
        x_corrupt_4 = self.n4(self.enc4(x_corrupt_3))
        x_corrupt_4 = self.relu(self.addNoise(x_corrupt_4, self.noise), self.prelu)
        x_corrupt_5 = self.n5(self.enc5(x_corrupt_4))
        x_corrupt_5 = self.relu(self.addNoise(x_corrupt_5, self.noise), self.prelu)
        #decode
        x_decode_4 = self.dec4(torch.cat((x_corrupt_4, x_corrupt_5), 1))
        x_decode_4 = self.dn4(x_decode_4)
        x_decode_3 = self.dec3(torch.cat((x_corrupt_3, x_decode_4), 1))
        x_decode_3 = self.dn3(x_decode_3)
        x_decode_2 = self.dec2(torch.cat((x_corrupt_2, x_decode_3), 1))
        x_decode_2 = self.dn2(x_decode_2)
        x_decode_1 = self.dec1(torch.cat((x_corrupt_1, x_decode_2),1 ))
        x_decode_1 = self.dn1(x_decode_1)
        #calculate distance between x_decode_i and x_clean_i
        d1 = ((x_decode_1 - x_clean_1)**2).mean()
        d2 = ((x_decode_2 - x_clean_2)**2).mean()
        d3 = ((x_decode_3 - x_clean_3)**2).mean()
        d4 = ((x_decode_4 - x_clean_4)**2).mean()
        return (F.log_softmax(x_clean_5))#, 0.1*(d1+d2+d3+d4))



def ladderNet(config):
    dropout = config.get('training', {}).get('dropout', '0.5')
    prelu = config.get('training', {}).get('PReLU', False)
    noise = config.get('training', {}).get('noise', '0.1')
    return LadderNet(dropout=dropout, prelu=prelu, noise=noise)

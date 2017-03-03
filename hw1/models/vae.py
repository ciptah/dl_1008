"""
Train a VAE model on unlabeled MNIST data.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

import logging

logger = logging.getLogger('vae')

MNIST_SIZE = 28*28
LOGPI = np.log(np.pi)
LOG2 = np.log(2)

def norm_pdf(mean, logvar, v):
    """Computes multivariate gaussian pdf. mean, logvar will be modified."""
    # Minibatch compatible.
    exponent = torch.sum(
            mean.sub_(v).abs_().pow_(2).div_(logvar.exp()).mul_(-0.5), 1)
    loglike = -0.5 * torch.sum(x_logvar.add_(LOGPI + LOG2), 1) + exponent
    return loglike

def kl_div_with_std_norm(mean, logvar):
    """KL divergence with standard multivariate normal. Consumes both inputs"""
    x = torch.sum(mean.mul_(mean).sub_(1).sub_(logvar), 1)
    tr = torch.sum(logvar.exp_(), 1)
    return tr.add_(x).div_(2)

class VAEDecoder(nn.Module):
    """The decoder generates X (an MNIST image) from z (latent variables).
    
    This is a simple feedforward NN that starts at k and ends at 784 (MNIST
    image size). The output is the mean and log-variance of the Gaussian,
    following Kingma's example."""

    def __init__(self, k):
        super(VAEDecoder, self).__init__()
        self.k = k
        self.fc1_size = 300
        self.fc2_size = 500
        
        self.fc1 = nn.Linear(k, self.fc1_size)
        self.fc2 = nn.Linear(k, self.fc2_size)
        self.mean = nn.Linear(k, MNIST_SIZE)
        self.logvar = nn.Linear(k, MNIST_SIZE)
        # Following Kingma's example we output log(var(x)) instead of var(x).

        logger.info('VAE Decoder: %d (z) -> %d -> %d -> %d (MNIST)',
                self.k, self.fc1_size, self.fc2_size, MNIST_SIZE)

    def forward(self, z):
        # Input is K-dim vector.
        h1 = F.relu(self.fc1(z))
        h2 = F.relu(self.fc2(h1))

        # No nonlinearities; preserve 
        mean = self.mean(h2)
        logvar = self.logvar(h2)
        return mean, logvar

class VAEEncoder(nn.Module):
    """The encoder infers the distribution of z (latent vars) for a given X.

    It outputs a mean and covariance for a Gaussian in z-space. The covariance
    is diagonal so only k values will be returned.

    The network itself is a ConvNet that follows BasicNet."""

    def __init__(self, k):
        super(VAEEncoder, self).__init__()
        self.k = k # Output

        # TODO: Configurable sizes/channels/parameters.
        conv1_channels = 10
        conv2_channels = 20

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5)
        self.fc1 = nn.Linear(320, k * 2)
        self.mean = nn.Linear(k * 2, k)
        self.logvar = nn.Linear(k * 2, k)

        logger.info('%d channels for 1st convnet', conv1_channels)
        logger.info('%d channels for 2nd convnet', conv2_channels)

    def forward(self, x):
        # Input is 1x28x28. Output is 2k.
        # N (minibatch size) is implicit.

        # Refer to basic_net.py for an explanation of this ConvNet.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x)), 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class VAE(nn.Module):
    """
    The VAE models MNIST with this process:
    - Sample z \in R^k from N(0, I) => I is the k-dim identity matrix
    - Feed z into a feedforward NN to get two R^(28*28) vectors \mu and \sigma^2
    - Sample X from N(\mu, \sigma^2)

    For training, the Q(z|x) is a Gaussian with some mean and diagonal
    covariance matrix. Given an X, get the mean and logvar by running X
    through a ConvNet based on basic_net.

    Hyperparameters:
    - k => Dimension of the latent variable z
    - \sigma => The variance of the decoder (samples X from z)

    The output is a pair of NNs: one for encoding and one for decoding.
    """

    def __init__(self, k=2):
        super(VAE, self).__init__()

        logger.info('VAE k=%d', k)

        self.encoder = VAEEncoder(k)
        self.decoder = VAEDecoder(k)

    def forward(self, x):
        # "Random seed" for z samples. They'll be multiplied with the
        # mean and variance to create a randomized sample (also known as
        # the reparametrization trick)
        random = torch.randn(x.size()[0], k)

        # Encode. (n, 768) -> (n, k)
        z_mean, z_logvar = self.encoder.forward(x)
        z_samples = random.mul_(z_logvar).add_(z_mean)

        # Decode. Given z, return distribution P(X|z). (n, k) -> (n, 768)
        x_mean, x_logvar = self.decoder.forward(z_samples)

        # E[log p(x|z)] - based on 1 sample of z. Logvar must be diagonal.
        # Intuitively if Q is good, then this should be close to P(X), so
        # getting this as high as possible is a good thing.
        log_x_z = norm_pdf(x_mean, x_logvar, x)

        # KL divergence - The bigger this is, the worse the score.
        # Intuitively, if diff is huge, it's not easy to trust the number above
        # so the lower bound widens.
        kl_div = kl_div_with_std_norm(z_mean, z_logvar)

        return log_x_z - kl_div


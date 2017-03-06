"""
Train a VAE model on unlabeled MNIST data.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from numpy import prod
from torch.autograd import Variable

import logging

logger = logging.getLogger('vae')

MNIST_SIZE = 28*28
C = -0.5 * np.log(2 * np.pi)

NORM_STD = 0.3081
NORM_MEAN = 0.1307
MIN = -NORM_MEAN / NORM_STD + 1e-4
MAX = (1-NORM_MEAN) / NORM_STD - 1e-4

def norm_pdf(mean, logvar, v):
    """Computes multivariate gaussian pdf."""
    z = (v - mean)
    exp = z * z * -0.5 / logvar.exp()
    res = exp + C + (-0.5 * logvar)
    return res.view(-1, MNIST_SIZE).sum(1)

def squared_error(mean, v):
    return -(mean - v).pow(2).view(-1, MNIST_SIZE).sum(1)

def cross_entropy(mean, v):
    mean = F.sigmoid(mean) + 1e-10
    v = (v - MIN) / (MAX - MIN)
    return (v * mean.log() + (1 - v) * (1 - mean).log()).view(-1, MNIST_SIZE).sum(1)

def logsumexp(x1, x2):
    return torch.log(torch.exp(x2) + x1)

def kl_div_with_std_norm(mean, logvar):
    """KL divergence with standard multivariate normal. Consumes both inputs"""
    return torch.squeeze(torch.sum(logvar.exp() + mean * mean - logvar - 1, 1) / 2)

class VAEDecoder(nn.Module):
    """The decoder generates X (an MNIST image) from z (latent variables).
    
    This is a simple feedforward NN that starts at k and ends at 784 (MNIST
    image size). The output is the mean and log-variance of the Gaussian,
    following Kingma's example."""

    def __init__(self, k, nonl=F.relu):
        super(VAEDecoder, self).__init__()
        self.k = k
        self.nonl = nonl

        self.fc = nn.Linear(k, 64 * 4 * 4)
        self.convt1 = nn.ConvTranspose2d(64, 16, 5, stride=2)
        self.convt2 = nn.ConvTranspose2d(16, 4, 4, stride=2)
        self.convt3 = nn.ConvTranspose2d(4, 1, 5)

        self.var_convt3 = nn.ConvTranspose2d(4, 1, 5)

    def forward(self, z):
        # Input is K-dim vector.
        h = self.nonl(self.fc(z))
        i = h.view(-1, 64, 4, 4)

        i = self.nonl(self.convt1(i))
        i = self.convt2(i)

        mean = self.convt3(i)
        logvar = self.var_convt3(i)
        return mean, logvar

class VAEEncoder(nn.Module):
    """The encoder infers the distribution of z (latent vars) for a given X.

    It outputs a mean and covariance for a Gaussian in z-space. The covariance
    is diagonal so only k values will be returned.

    The network itself is a ConvNet that follows BasicNet."""

    def __init__(self, k, nonl=F.relu, use_convnets=True):
        super(VAEEncoder, self).__init__()
        self.k = k # Output
        self.nonl = nonl
        self.fc_dim = 1024

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=5)
        self.fc = nn.Linear(64 * 4 * 4, self.fc_dim)

        self.mean = nn.Linear(self.fc_dim, k)
        self.logvar = nn.Linear(self.fc_dim, k)

    def forward(self, x):
        x = self.nonl(self.conv1(x))
        x = self.nonl(F.max_pool2d(self.conv2(x), 2))
        x = self.nonl(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        x = self.nonl(self.fc(x))

        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar


class VAE(nn.Module):
    """
    The VAE models MNIST with this process:
    - Sample z \in R^k from N(0, I) => I is the k-dim identity matrix
    - Feed z into a feedforward NN to get two R^(28*28) vectors \mu and \sigma^2
    - Sample X from N(\mu, \sigma^2)

    This is equivalent to the M1 variant of Kingma et al. 2014.

    For training, the Q(z|x) is a Gaussian with some mean and diagonal
    covariance matrix. Given an X, get the mean and logvar by running X
    through a ConvNet based on basic_net.

    Hyperparameters:
    - k => Dimension of the latent variable z

    The output is a pair of NNs: one for encoding and one for decoding.
    """

    def __init__(self, config):
        super(VAE, self).__init__()
        vae_config = config.get('vae', {})
        train_config = config.get('training', {})
        logger.info('starting new model from scratch.')

        # Optimization parameters.
        # - Number of real-valued latent variables.
        self.k = vae_config.get('num_latent_vars', 2)
        # - nonlinear function to use.
        self.nonl = get_nonl(train_config.get('nonlinearity', 'relu'))
        # - which scoring function to use. "Gaussian PDF" is the actual
        #   "correct" one, the rest are just hacks.
        self.score_fn = vae_config.get('score_fn', 'gaussian_pdf')

        self.counter = 0

        self.encoder = VAEEncoder(self.k, nonl=self.nonl)
        self.decoder = VAEDecoder(self.k, nonl=self.nonl)

        self.update(config, vae_config)
        self.diagnostics()

    def update(self, config, vae_config):
        # - Standard deviation of Gaussian to convert to a z-sample.
        self.epsilon = vae_config.get('sampling_epsilon', 1.0)
        # - Multiply KL divergence by this number before computing total cost.
        #   Acts as a regularization parameter.
        self.kl_multiplier = vae_config.get('kl_multiplier', 1.0)
        # - whether to use/update conv nets during encoding.
        self.use_convnets = vae_config.get('use_convnets', True)
        self.encoder.use_convnets = self.use_convnets
        # - Added variance to the gaussian.
        self.add_stdev = vae_config.get('add_stdev', 0.1)

    def param_count(self):
        return sum([prod(w.size()) for w in self.parameters()])

    def diagnostics(self):
        logger.info('model:VAE k=%d', self.k)
        logger.info('model:epsilon=%f', self.epsilon)
        logger.info('model:nonlinearity=%s', self.nonl)
        logger.info('model:kl_multiplier=%f', self.kl_multiplier)
        logger.info('model:score_fn=%s', self.score_fn)
        logger.info('model:param_count=%d', self.param_count())
        logger.info('model:add_stdev=%f', self.add_stdev)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def score(self, mean, logvar, x):
        if self.score_fn == 'squared_error':
            return squared_error(mean, x)
        elif self.score_fn == 'bce':
            return cross_entropy(mean, x)
        else:
            return norm_pdf(mean, logvar, x)

    def forward(self, x):
        """Forward step decodes and reconstructs x (the data), then
        returns the VAE loss and the reconstructed means.
        """
        # Encode. (n, 768) -> (n, k)
        z_mean, z_logvar = self.encode(x)
        random = Variable(torch.randn(x.size()[0], self.k)) * self.epsilon
        z_samples = z_mean + random * (z_logvar / 2).exp()

        # Decode. Given z, return distribution P(X|z). (n, k) -> (n, 768)
        x_mean, x_logvar = self.decode(z_samples)
        x_logvar = logsumexp(self.add_stdev, x_logvar)

        score = self.score(x_mean, x_logvar, x) # Estimate for P(x)
        kl_div = kl_div_with_std_norm(z_mean, z_logvar)

        if self.counter % 100 == 0:
            logger.debug('logP: %f   KL: %f', score.data.mean(), kl_div.data.mean())
            logger.debug('X: mean: %f   std: %s', x.data.mean(), x.data.std())
            logger.debug('S: mean: %f   std: %s', x_mean.data.mean(), x_mean.data.std())
            logger.debug('Z: mean: %f   std: %s', z_samples.data.mean(), z_samples.data.std())
        self.counter += 1

        # The loss is the negative of the score.
        return -(score - self.kl_multiplier * kl_div), x_mean

class VAETrainer:
    """Trains VAE with unlabeled data.

    This is different from predictive model because the output isn't a
    classifier, but the encoder/decoder.
    """
    def __init__(self, config):
        self.config = config
        vae_config = config.get('vae', {})
        train_config = config.get('training')

        # Adam parameters.
        # Get learning rate from vae block, otherwise default to global rate
        learning_rate = vae_config.get('learning_rate', 
                train_config.get('learning_rate', 3e-4))
        weight_decay = vae_config.get('weight_decay', 
                train_config.get('weight_decay', 0.0))

        # Reporting parameters.
        # - Number of epochs before getting a new comparison image.
        self.image_per = vae_config.get('image_per', 1)
        # - Where to save the image file.
        self.image_file_name = vae_config.get('image_file_name', 'vae.png')
        # - Where to save the VAE model.
        self.model_file_name = vae_config.get('model_file_name', 'vae.torch')

        if config.get('vae', {}).get('continue', False):
            logger.info('continuing from previous model.')
            model_file_name = config.get('vae').get('model_file_name', 'vae.torch')
            logger.info('loading model from %s', model_file_name)
            self.model = torch.load(model_file_name)
            self.model.update(config, vae_config)
        else:
            self.model = VAE(config)

        self.optimizer = optim.Adam(self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

        # This is a list of (x, reconstruction)
        self.history = []
        self.last_minibatch = None

    def start_train(self):
        if self.config.get('vae', {}).get('continue', False):
            self.model.train()

    def train_batch(self, data, batch_idx=0):
        """Where the *~*~magic~*~* happens."""
        self.optimizer.zero_grad()
        loss, reconstructed_data = self.model(data)
        minibatch_loss = loss.mean()
        minibatch_loss.backward() # Specifically, it's here.
        self.optimizer.step()

        # We need to clamp the reconstructed image so it doesn't overflow.
        reconstructed_image = reconstructed_data.clamp(MIN, MAX)
        self.last_minibatch = (minibatch_loss.data[0], data.data, reconstructed_image.data)
        return minibatch_loss.data[0]

    def epoch_done(self, epoch_id):
        loss, data, reconst = self.last_minibatch
        diff = (data - reconst)
        self.history.append((data, reconst, diff * diff))
        self.last_minibatch = None

    def training_done(self):
        def denorm(x):
            return x.mul(NORM_STD).add(NORM_MEAN).clamp(MIN, MAX)
        images_per_epoch = 10
        tensors = []
        for i, epoch_data in enumerate(self.history[-50:]):
            data, reconst, diff = epoch_data
            if i % self.image_per != 0:
                continue
            tensors.append(denorm(data[:images_per_epoch]))
            tensors.append(denorm(reconst[:images_per_epoch]))
        res = torch.cat(tensors)
        logger.debug('Image grid size: %s', res.size())
        torchvision.utils.save_image(res, self.image_file_name, nrow=images_per_epoch)
        torch.save(self.model, self.model_file_name)
        logger.debug('Saved model to %s', self.model_file_name)

        logger.info('this model has gone through %d minibatches', self.model.counter)

def get_nonl(config):
    return {
        'relu': F.relu,
        'prelu': F.prelu,
        'tanh': F.tanh,
        'softplus': F.softplus
    }[config]

def vae(config):
    return VAETrainer(config)


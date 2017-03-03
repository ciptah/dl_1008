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
from torch.autograd import Variable

import logging

logger = logging.getLogger('vae')

MNIST_SIZE = 28*28
C = -0.5 * np.log(2 * np.pi)

def norm_pdf(mean, logvar, v):
    """Computes multivariate gaussian pdf."""
    z = (v - mean)
    exp = z * z * -0.5 / logvar.exp()
    return torch.sum(exp + C + (-0.5 * logvar), 1)

def mse(m, unused_, v):
    # Negative so higher = better
    return 1/(1e-10 + torch.mean((v - m).pow(2), 1))

gen_losses = {
    'norm_pdf': norm_pdf,
    'mse': mse
}

def kl_div_with_std_norm(mean, logvar):
    """KL divergence with standard multivariate normal. Consumes both inputs"""
    return torch.sum(logvar.exp() + mean * mean - logvar - 1, 1) / 2

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

class VAEDecoder(nn.Module):
    """The decoder generates X (an MNIST image) from z (latent variables).
    
    This is a simple feedforward NN that starts at k and ends at 784 (MNIST
    image size). The output is the mean and log-variance of the Gaussian,
    following Kingma's example."""

    def __init__(self, k, nonl=F.relu):
        super(VAEDecoder, self).__init__()
        self.nonl = nonl
        self.k = k
        self.fc_sizes = [1200]
       
        prev_dim = k
        self.fcs = []
        for i, sz in enumerate(self.fc_sizes):
            fc = nn.Linear(prev_dim, sz)
            setattr(self, 'fc_{}'.format(i), fc)
            self.fcs.append(fc)
            prev_dim = sz

        self.mean = nn.Linear(prev_dim, MNIST_SIZE)
        self.logvar = nn.Linear(prev_dim, MNIST_SIZE)
        # Following Kingma's example we output log(var(x)) instead of var(x).

        logger.info('decoder: %d (z) -> %s -> %d (MNIST)',
                self.k, self.fc_sizes, MNIST_SIZE)

    def forward(self, z):
        # Input is K-dim vector.
        h = z
        for fc in self.fcs:
            h = self.nonl(fc(h))

        # No nonlinearities; preserve 
        mean = self.mean(h)
        logvar = self.logvar(h)
        return mean, logvar

class VAEEncoder(nn.Module):
    """The encoder infers the distribution of z (latent vars) for a given X.

    It outputs a mean and covariance for a Gaussian in z-space. The covariance
    is diagonal so only k values will be returned.

    The network itself is a ConvNet that follows BasicNet."""

    def __init__(self, k, nonl=F.relu):
        super(VAEEncoder, self).__init__()
        self.k = k # Output
        self.nonl = nonl

        # TODO: Configurable sizes/channels/parameters.
        conv1_channels = 10
        conv2_channels = 20
        self.fc_sizes = [1200]

        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=5)
        prev_dim = 320
        self.fcs = []
        for i, sz in enumerate(self.fc_sizes):
            fc = nn.Linear(prev_dim, sz)
            setattr(self, 'fc_{}'.format(i), fc)
            self.fcs.append(fc)
            prev_dim = sz
        self.mean = nn.Linear(prev_dim, k)
        self.logvar = nn.Linear(prev_dim, k)

        logger.info('encoder:%d channels for 1st convnet', conv1_channels)
        logger.info('encoder:%d channels for 2nd convnet', conv2_channels)

    def forward(self, x):
        # Input is NBatch x 1x28x28. Output is 2k.
        # N (minibatch size) is implicit.
        #logger.debug(x.size())

        # Refer to basic_net.py for an explanation of this ConvNet.
        x = self.nonl(F.max_pool2d(self.conv1(x), 2))
        x = self.nonl(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        for fc in self.fcs:
            x = self.nonl(fc(x))
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

    def __init__(self, k=2, epsilon=1.0, kl_multiplier=1.0, nonl=F.relu):
        super(VAE, self).__init__()

        logger.info('model:VAE k=%d', k)
        logger.info('model:epsilon=%f', epsilon)
        logger.info('model:nonl=%s', nonl)
        
        self.k = k
        self.epsilon = epsilon
        self.kl_multiplier = kl_multiplier
        self.encoder = VAEEncoder(k, nonl=nonl)
        self.decoder = VAEDecoder(k, nonl=nonl)

        self.counter = 0

    def forward(self, x, gen_loss=norm_pdf):
        """Forward step decodes and reconstructs x (the data), then
        returns the VAE loss and the reconstructed means.
        """
        # "Random seed" for z samples. They'll be multiplied with the
        # mean and variance to create a randomized sample (also known as
        # the reparametrization trick)
        random = Variable(torch.randn(x.size()[0], self.k)) * self.epsilon

        # Encode. (n, 768) -> (n, k)
        z_mean, z_logvar = self.encoder(x)
        z_samples = z_mean + random * (z_logvar / 2).exp()

        # Decode. Given z, return distribution P(X|z). (n, k) -> (n, 768)
        x_mean, x_logvar = self.decoder(z_samples)

        # E[log p(x|z)] - based on 1 sample of z. Logvar must be diagonal.
        # Intuitively if Q is good, then this should be close to P(X), so
        # getting this as high as possible is a good thing.
        log_x_z = gen_loss(x_mean, x_logvar, x.view(-1, MNIST_SIZE))

        # KL divergence - The bigger this is, the worse the score.
        # Intuitively, if diff is huge, it's not easy to trust the number above
        # so the lower bound widens.
        kl_div = kl_div_with_std_norm(z_mean, z_logvar)

        self.counter += 1
        if self.counter % 100 == 0:
            logger.debug('logP: %f   KL: %f', log_x_z.data.mean(), kl_div.data.mean())
            logger.debug('X: mean: %f   std: %s', x.data.mean(), x.data.std())
            logger.debug('S: mean: %f   std: %s', x_mean.data.mean(), x_mean.data.std())
            logger.debug('Z: mean: %f   std: %s', z_samples.data.mean(), z_samples.data.std())

        # The loss is the negative of the score.
        return -(log_x_z - self.kl_multiplier * kl_div), x_mean

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

        # Optimization parameters.
        # - Number of real-valued latent variables.
        z_size = vae_config.get('num_latent_vars', 2)
        # - Standard deviation of Gaussian to convert to a z-sample.
        epsilon = vae_config.get('sampling_epsilon', 1.0)
        # - Multiply KL divergence by this number before computing total cost.
        #   Acts as a regularization parameter.
        self.kl_multiplier = vae_config.get('kl_multiplier', 1.0)
        # - nonlinear function to use.
        self.nonl = get_nonl(train_config.get('nonlinearity', 'relu'))

        # Reporting parameters.
        # - Number of epochs before getting a new comparison image.
        self.image_per = vae_config.get('image_per', 1)
        # - Where to save the image file.
        self.image_file_name = vae_config.get('image_file_name', 'vae.png')
        # - Where to save the VAE model.
        self.model_file_name = vae_config.get('model_file_name', 'vae.torch')
        # - Which generation loss to use (gaussian pdf vs mean squared error)
        gen_loss_str = vae_config.get('generation_loss', 'norm_pdf')
        self.gen_loss = gen_losses[gen_loss_str]

        logger.info('trainer:num_latent_vars: %d', z_size)

        self.model = VAE(z_size, epsilon, self.kl_multiplier)
        self.optimizer = optim.Adam(self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay)

        # This is a list of (x, reconstruction)
        self.history = []
        self.last_minibatch = None

    def start_train(self):
        self.model.train()

    def train_batch(self, data, batch_idx=0):
        """Where the *~*~magic~*~* happens."""
        self.optimizer.zero_grad()
        loss, reconstructed_data = self.model(data, gen_loss=self.gen_loss)
        minibatch_loss = loss.mean()
        minibatch_loss.backward() # Specifically, it's here.
        self.optimizer.step()

        reconstructed_image = reconstructed_data.view_as(data)
        self.last_minibatch = (minibatch_loss.data[0], data.data, reconstructed_image.data)
        return minibatch_loss.data[0]

    def epoch_done(self, epoch_id):
        if epoch_id % self.image_per == 0:
            self.history.append(self.last_minibatch)
            self.last_minibatch = None

    def training_done(self):
        images_per_epoch = 10
        tensors = []
        for epoch_data in self.history:
            mloss, data, reconst = epoch_data
            tensors.append(data[:images_per_epoch,:,:,:])
            tensors.append(reconst[:images_per_epoch,:,:,:])
        res = torch.cat(tensors)
        logger.debug('Image grid size: %s', res.size())
        torchvision.utils.save_image(res, self.image_file_name, nrow=images_per_epoch)
        torch.save(self, self.model_file_name)

def get_nonl(config):
    return {
        'relu': F.relu,
        'prelu': F.prelu,
        'tanh': F.tanh,
        'softplus': F.softplus
    }[config]

def vae(config):
    return VAETrainer(config)


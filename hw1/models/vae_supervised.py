"""
Use a trained VAE model to expand a small training set with more examples.
"""
import torch
import random
import logging
import sys

from torch.autograd import Variable

logger = logging.getLogger('vae_expander')

NORM_STD = 0.3081
NORM_MEAN = 0.1307
MIN = -NORM_MEAN / NORM_STD + 1e-4
MAX = (1-NORM_MEAN) / NORM_STD - 1e-4

class VAEExpander:
    def __init__(self, config, vae_model, batch_size=64):
        xp_config = config.get('vae_expander', {})

        self.p = xp_config.get('p', 0.7) # Probability of replacement

        self.vae_model = vae_model

    def build(self, train_loader):
        self.train_loader = train_loader
        self.by_digit = [[] for i in range(10)]
        self.dataset = train_loader.dataset

        for data, target in self.train_loader:
            vdata = Variable(data)
            latent_data, unused_ = self.vae_model.encode(vdata)
            self.vae_model.decode(latent_data)
            for example, latent, label in zip(data.split(1, 0),
                    latent_data.data.split(1, 0), target.split(1, 0)):
                self.by_digit[label[0]].append((example, latent))

        logger.info('split results: %s', [len(x) for x in self.by_digit])

    def get_hull_vectors(self, digits):
        # Idea: for each example, pick two random examples, pick a point
        # in the line segment between those two examples.
        examples = []
        for d in digits:
            p1 = random.choice(self.by_digit[d])
            p2 = random.choice(self.by_digit[d])
            point = random.random()
            z = point * p1[1] + (1 - point) * p2[1]
            examples.append(z)
        z_batch = torch.cat(examples)
        new_data, unused_ = self.vae_model.decode(Variable(z_batch))
        return new_data.data

    def hull_mode(self):
        for data, target in self.train_loader:
            split = data.split(1, 0)
            mask = torch.bernoulli(torch.ones(len(split)).mul_(self.p))
            new_data = self.get_hull_vectors(target)
            mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(new_data)
            data = (new_data * mask) + (data * (1 - mask))
            yield data, target

    def __len__(self):
        return len(self.train_loader)

    def __iter__(self):
        # Impersonate a Data Loader.
        # This is slow, but can act as a plug-in replacement for the data provider.
        return self.hull_mode()

def augment(config, loader):
    vae = torch.load(config['vae_expander']['model_file_name'])
    vaes = VAEExpander(config, vae)
    vaes.build(loader)
    return vaes


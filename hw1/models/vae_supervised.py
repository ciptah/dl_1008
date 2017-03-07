"""
Use a trained VAE model to expand a small training set with more examples.
"""
import torch
import random
import logging
import sys
import pickle
import numpy as np
import data_provider as dp

from torch.autograd import Variable
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

logger = logging.getLogger('vae_expander')

NORM_STD = 0.3081
NORM_MEAN = 0.1307
MIN = -NORM_MEAN / NORM_STD + 1e-4
MAX = (1-NORM_MEAN) / NORM_STD - 1e-4

class HullSampler:
    """Represents a convex hull of a collection of points in the latent space.

    The hull is built on a lower-dimensional representation using PCA.
    Then points are sampled using rejection sampling.
    This gets slow very quickly as you ramp up the number of dimensions.
    """

    def __init__(self, points, reduce_dim=6, threshold=0.0):
        # Points = (N x d) numpy matrix
        points = np.array([x[1].numpy() for x in points])
        points = np.resize(points, (points.shape[0], np.prod(points.shape[1:])))

        self.dim = reduce_dim
        self.threshold = threshold
        self.pca = PCA(reduce_dim, whiten=True)
        npoints = self.pca.fit_transform(points)
        logger.debug(self.pca.explained_variance_)

        self.hull = ConvexHull(npoints)

        # Find the bounding box.
        self.bbox_min = np.min(npoints, axis=0)
        self.bbox_max = np.max(npoints, axis=0)
        logger.debug('min: %s', self.bbox_min)
        logger.debug('max: %s', self.bbox_max)

        bbox_vol = np.prod(self.bbox_max - self.bbox_min)
        hull_vol = self.hull.volume
        logger.debug('hull volume: %f / %f (%f)',
                hull_vol, bbox_vol, hull_vol / bbox_vol)

        self.hit = 0
        self.tries = 0
        self.batches = 0

    def hit_rate(self):
        if self.tries == 0:
            return 0.0
        return self.hit / self.tries

    def sampler(self):
        batchsize = 4096 # to speed up performance
        sample = torch.Tensor(batchsize, self.dim)
        min_val = torch.Tensor(self.bbox_min).unsqueeze(0).expand_as(sample)
        max_val = torch.Tensor(self.bbox_max).unsqueeze(0).expand_as(sample)
        eqns = torch.Tensor(self.hull.equations[:,:-1])
        limits = torch.Tensor(self.hull.equations[:, -1])
        limits = limits.unsqueeze(1).expand(limits.size()[0], batchsize)
        while True:
            self.batches += 1
            sample.uniform_(0, 1).mul_(max_val - min_val).add_(min_val)
            latents = self.pca.inverse_transform(sample.numpy())
            test = eqns.mm(sample.t()) + limits
            test = test.max(dim=0)[0].squeeze() # Check each example
            for i in range(batchsize):
                self.tries += 1
                if test[i] < self.threshold:
                    self.hit += 1
                    yield torch.Tensor(latents[i,:]).view(1, 16)

class VAEExpander(dp.Loader):
    def __init__(self, config, vae_model, train_loader):
        super(VAEExpander, self).__init__(None)
        xp_config = config.get('vae_expander', {})

        # Parameters:
        # - how to sample. Values: 'line', 'hull'
        self.mode = xp_config.get('mode', 'hull')
        # - probability that sample is randomized.
        self.p = xp_config.get('p', 0.7)
        # - how far "outside" the line segment between two points to sample.
        self.extra = xp_config.get('extra', 1.5)
        # - how many times to multiply the epoch size.
        self.epoch_multiple = xp_config.get('epoch_multiple', 10)
        # - dimension to reduce to before computing convex hull.
        self.hull_dim = xp_config.get('hull_dim', 6)
        # - at what (approximate) epoch to activate.
        self.activate_at = xp_config.get('activate_at', 50)
        # - when in mixed mode, what percentage of randomized examples
        #   are from the line generator.
        self.mixed_mode_line_proportion = xp_config.get('mixed_mode_line_proportion', 0.5)

        self.vae_model = vae_model
        self.epoch_counter = 0

        self.build(train_loader)

    def build(self, train_loader):
        self.train_loader = train_loader
        self.by_digit = [[] for i in range(10)]

        logger.info('sorting and building latent vectors.')
        for data, target in self.train_loader:
            vdata = Variable(data)
            latent_data, unused_ = self.vae_model.encode(vdata)
            self.vae_model.decode(latent_data)
            for example, latent, label in zip(data.split(1, 0),
                    latent_data.data.split(1, 0), target.split(1, 0)):
                self.by_digit[label[0]].append((example, latent))

        logger.info('split results: %s', [len(x) for x in self.by_digit])

        if self.mode == 'hull' or self.mode == 'mixed':
            logger.info('calculating samplers...')
            self.hulls = [HullSampler(points, self.hull_dim, self.extra)
                    for points in self.by_digit]
            self.hull_samplers = [s.sampler() for s in self.hulls]

        logger.info('ready!')

    def print_info(self):
        if self.mode == 'hull' or self.mode == 'mixed':
            for i, hull in enumerate(self.hulls):
                logger.debug('Class %d: hit rate %f (%d/%d), %d batches',
                        i, hull.hit_rate(), hull.hit, hull.tries, hull.batches)

    def get_random_from_line(self, d):
        p1 = random.choice(self.by_digit[d])
        p2 = random.choice(self.by_digit[d])
        point = random.random() * (1 + 2 * self.extra) - self.extra
        return point * p1[1] + (1 - point) * p2[1]

    def get_random_from_hull(self, d):
        return next(self.hull_samplers[d])

    def get_random_mixed(self, d):
        if random.random() < self.mixed_mode_line_proportion:
            return self.get_random_from_line(d)
        else:
            return self.get_random_from_hull(d)

    def get_random_samples(self, digits):
        # Idea: for each example, pick two random examples, pick a point
        # in the line segment between those two examples.
        examples = []
        for d in digits:
            if self.mode == 'hull':
                z = self.get_random_from_hull(d)
            elif self.mode == 'line':
                z = self.get_random_from_line(d)
            elif self.mode == 'mixed':
                z = self.get_random_mixed(d)
            examples.append(z)
        z_batch = torch.cat(examples)
        new_data, unused_ = self.vae_model.decode(Variable(z_batch))
        return new_data.data

    def mix_samples(self, data, target):
        split = data.split(1, 0)
        mask = torch.bernoulli(torch.ones(len(split)).mul_(self.p))
        new_data = self.get_random_samples(target)
        mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(new_data)
        data = (new_data * mask) + (data * (1 - mask))
        return data, target

    def __len__(self):
        return len(self.train_loader) * self.epoch_multiple

    def example_count(self):
        return self.train_loader.example_count() * self.epoch_multiple

    def __iter__(self):
        # Impersonate a Data Loader.
        # This is slow, but can act as a plug-in replacement for the data provider.
        self.epoch_counter += 1
        active = False
        if self.epoch_counter == self.activate_at:
            logger.warn('activating VAE generator')
        if self.epoch_counter >= self.activate_at:
            active = True
        for multiple in range(self.epoch_multiple):
            for data, target in self.train_loader:
                if active:
                    yield self.mix_samples(data, target)
                else:
                    yield data, target

class Mixer(dp.Loader):
    """Pools multiple VAEExpanders together and calls them in alternate."""
    def __init__(self, models):
        super(Mixer, self).__init__(None)
        self.models = models
        logger.debug('VAE mixer with %d models, size %d', len(models), len(self))

    def __iter__(self):
        iters = [iter(model) for model in self.models]
        while True:
            has_next = False
            random.shuffle(iters)
            for i in iters:
                try:
                    yield next(i)
                    has_next = True
                except StopIteration:
                    pass
            if not has_next:
                return

    def __len__(self):
        return sum(len(m) for m in self.models)

    def example_count(self):
        return sum(m.example_count() for m in self.models)

    def print_info(self):
        for m in self.models:
            m.print_info()

def augment(config, loader, model_override=None):
    if model_override:
        mfilename = model_override
    else:
        mfilename = config['vae_expander']['model_file_name']
    if isinstance(mfilename, list):
        logger.info('mixing multiple VAE models.')
        vaes = [augment(config, loader, mf) for mf in mfilename]
        return Mixer(vaes)
    else:
        vae = pickle.load(open(mfilename, 'rb'))
        vaes = VAEExpander(config, vae, loader)
        return vaes


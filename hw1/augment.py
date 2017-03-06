"""Augments training data with some quick but helpful transforms.
"""

import logging
import numpy as np
import data_provider as dp

from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage

logger = logging.getLogger('default_augmentation')

def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def preprocess_img(img, width, height, limits_ranslate=(-3, 3), limits_rotate=(-20, 20)):
    # add noise
    img += np.random.normal(scale=0.2, size=(width, height))

    # translation
    dx = np.random.choice(range(limits_ranslate[0], limits_ranslate[1] + 1))
    dy = np.random.choice(range(limits_ranslate[0], limits_ranslate[1] + 1))
    img[:, :] = ndimage.interpolation.shift(img, (dx, dy))

    # rotation
    degree = np.random.choice(range(limits_rotate[0], limits_rotate[1] + 1))
    ndimage.rotate(img, degree, reshape=False, output=img[:, :])

    # elastic distrotion
    img[:, :] = elastic_transform(img, 36, 8)
    # return img_arr


def default_augmentation(data):
    # Data is a FloatTensor (not a Variable<FloatTensor>)
    batch_num = data.size()[0]
    width = data.size()[2]
    height = data.size()[3]

    data = data.numpy()
    # output = np.ones((batch_num,1,width,height))
    for i in range(0, batch_num):
        # output[i,0,:,:] = preprocess_img(data[i,0,:,:], width, height)
        preprocess_img(data[i, 0, :, :], width, height)
        # return Variable(torch.Tensor(output))

class DefaultAugmenter(dp.Loader):
    def __init__(self, loader):
        super(DefaultAugmenter, self).__init__(None)
        self.loader = loader

    def __iter__(self):
        for data, target in self.loader:
            default_augmentation(data)
            yield data, target

    def __len__(self):
        return len(self.loader)

    def example_count(self):
        return self.loader.example_count()

def create_augmenter(config, loader):
    logger.info('using default augmentation')
    return DefaultAugmenter(loader)

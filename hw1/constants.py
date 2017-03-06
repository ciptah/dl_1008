import augment
import models.vae_supervised as vae_supervised

from models.basic_net import basicNet
from models.ladder_net import ladderNet
from models import pseudo_label
from models.SWWAE import swwae
from models.vae import vae

# The values of this dict are functions that take the config object and returns
# a subclass of nn.Module.
models = {
    'basic': basicNet,
    'ladder': ladderNet,
    'swwae': swwae,
}

psuedo_label_func_dict = {
    "default" : pseudo_label.default_pseudo_label_func
}

# What model to run on unlabeled data.
unlabeled_models = {
    'vae': vae
}

def chain_augmenter(config, loader):
    vae_loader = vae_supervised.augment(config, loader)
    return augment.create_augmenter(config, vae_loader)

def no_augment(config, loader):
    return loader

# Various ways to augment training data.
# Method takes the configuration object and the training data loader.
augment_training = {
    'vae': vae_supervised.augment,
    'default': augment.create_augmenter,
    'chain': chain_augmenter,
    'none': no_augment
}


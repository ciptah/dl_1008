from models.basic_net import basicNet
from models.vae import vae
import models.vae_supervised as vae_supervised

# The values of this dict are functions that take the config object and returns
# a subclass of nn.Module.
models = {
    'basic': basicNet
}

unlabeled_models = {
    'vae': vae
}

# Various ways to augment training data.
# Method takes the configuration object and the training data loader.
augment_training = {
    'vae': vae_supervised.augment
}

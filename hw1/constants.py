from models.basic_net import basicNet
from models.vae import vae

# The values of this dict are functions that take the config object and returns
# a subclass of nn.Module.
models = {
    'basic': basicNet
}

unlabeled_models = {
    'vae': vae
}

from models.basic_net import basicNet
from models import pseudo_label
from models.SWWAE import swwae

# The values of this dict are functions that take the config object and returns
# a subclass of nn.Module.
models = {
    'basic': basicNet,
    'swwae': swwae
}

psuedo_label_func_dict = {
    "default" : pseudo_label.default_pseudo_label_func
}

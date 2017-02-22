from models.basic_net import basicNet

# The values of this dict are functions that take the config object and returns
# a subclass of nn.Module.
models = {
    'basic': basicNet
}

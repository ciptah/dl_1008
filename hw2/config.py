import logging
import json
from datetime import datetime
import itertools

logger = logging.getLogger('config')

date = datetime.now().strftime('%Y%m%d-%H%M%S')

# parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
# parser.add_argument('--model', type=str, default='LSTM',
#                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
# parser.add_argument('--emsize', type=int, default=50,
#                     help='size of word embeddings')
# parser.add_argument('--nhid', type=int, default=50,
#                     help='humber of hidden units per layer')
# parser.add_argument('--nlayers', type=int, default=1,
#                     help='number of layers')
# parser.add_argument('--lr', type=float, default=20,
#                     help='initial learning rate')
# parser.add_argument('--clip', type=float, default=0.5,
#                     help='gradient clipping')
# parser.add_argument('--epochs', type=int, default=6,
#                     help='upper epoch limit')
# parser.add_argument('--batch-size', type=int, default=20, metavar='N',
#                     help='batch size')
# parser.add_argument('--bptt', type=int, default=20,
#                     help='sequence length')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--cuda', action='store_true',
#                     help='use CUDA')
# parser.add_argument('--log-interval', type=int, default=200, metavar='N',
#                     help='report interval')
# parser.add_argument('--save', type=str,  default='model.pt',
#                     help='path to save the final model')

def single_setter(key, value):
    def fn(x):
        x[key] = value
    return fn

# Generate actual configurations from the template.
def generate_configs(template, max_runs=1000):
    # Each item in setters corresponds to a key (clip, nlayers, etc.)
    # Each item is a list of possible values
    setters = []
    for key, value in template.items():
        if isinstance(value, list):
            options = []
            for val in value:
                options.append(single_setter(key, val))
            setters.append(options)
        else:
            setters.append([single_setter(key, value)])

    for id_num, template in enumerate(itertools.product(*setters)):
        if id_num > max_runs:
            return
        config = {}
        experiment_id = '{}-{:02d}'.format(date, id_num)
        for setter in template:
            setter(config)
        config['experiment_id'] = experiment_id
        config['logfile'] = config['logfile'].format(experiment_id)
        config['configfile'] = config['configfile'].format(experiment_id)
        config['save'] = config['save'].format(experiment_id)
        logger.debug('CONFIGURATION: %s', json.dumps(config, indent=2))
        yield config

def build_config_template(config_filename):
    """
    Generates a config template.
    """
    with open(config_filename, 'r') as f:
        config_json = json.load(f)

    logger.info('Loaded JSON config from %s', config_filename)

    if config_json.get('verbose', False):
        logging.basicConfig(level=logging.DEBUG)
        logger.debug('Turning on verbose logging.')
    else:
        # Default log level is info.
        logging.basicConfig(level=logging.INFO)

    logger.info('CONFIGURATION: %s', json.dumps(config_json, indent=2))
    return config_json


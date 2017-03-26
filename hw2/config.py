import logging
import json
from datetime import datetime
import itertools

logger = logging.getLogger('config')

date = datetime.now().strftime('%Y%m%d-%H%M%S')

def single_setter(key, value):
    def fn(x):
        x[key] = value
    return fn

# Generate actual configurations from the template.
def generate_configs(template):
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
        config = {}
        experiment_id = '{}-{:02d}'.format(date, id_num)
        for setter in template:
            setter(config)
        config['experiment_id'] = experiment_id
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


import logging
import json

def build_config(config_filename, logger):
    """
    Function that builds a config object from file
    :param config_filename:
    :param logger:
    :return:
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

    logger.debug('CONFIGURATION: %s', json.dumps(config_json, indent=2))
    return config_json


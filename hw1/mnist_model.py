import sys
import json
import logging

logger = logging.getLogger('main')

if len(sys.argv) < 2:
    print('Need a config filename!')
    sys.exit(1)

config_filename = sys.argv[1]
with open(config_filename, 'r') as f:
    config_json = json.load(f)

if 'verbose' in config_json and config_json['verbose']:
    logging.basicConfig(level=logging.DEBUG)
    logger.debug('Turning on verbose logging.')
else:
    # Default log level is info.
    logging.basicConfig(level=logging.INFO)

logger.info('Loaded JSON config from %s', config_filename)
logger.debug('CONFIGURATION: %s', json.dumps(config_json, indent=2))

# Parameters are loaded, now use them to read the data.

# Do some training.

# Output the model file.

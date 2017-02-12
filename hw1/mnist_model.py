import sys
import json
import logging

logger = logging.getLogger('main')

if len(sys.argv) < 2:
    logger.warn('Need a config filename!')
    sys.exit(1)

if len(sys.argv) >= 3 and sys.argv[2] == 'debug':
    logging.basicConfig(level=logging.DEBUG)

config_filename = sys.argv[1]
logger.info('Loading JSON config from %s', config_filename)
with open(config_filename, 'r') as f:
    config_json = json.load(f)

logger.debug('CONFIGURATION: %s', json.dumps(config_json, indent=2))

# Parameters are loaded, now use them to read the data.

# Do some training.

# Output the model file.

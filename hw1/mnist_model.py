import sys
import json
import logging

if len(sys.argv) < 2:
    logging.warn('Need a config filename!'

config_filename = sys.argv[1]
logging.info('Loading JSON config from %s', config_filename)
with open(config_filename, 'r') as f:
    config_json = json.load(config_filename)

logging.debug('Config: %s', json.dumps(config_json))

"""Main program entry point."""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

from config import build_config

# initialization
logger = logging.getLogger('main')


def training(config, config_training):
    """Training module."""
    # Num of epochs
    num_epochs = config_training.get('num_epochs', 10)
    reporting_period = config_training.get('report_every_n_batch', 100)
    save_dir = config_training.get('save_dir', './data/model/')
    model_name = config_training.get('model', 'basic')

    logger.info('Using model %s', model_name)
    logger.info('Saving model to %s/%s.p', save_dir, model_name)

    # # # TRAINING CODE GOES HERE # # #

    logger.info('running for %d epochs', num_epochs)


def main(config_filename):
    # parse config
    config = build_config(config_filename, logger)

    for phase in config['phases']: # Mandatory arg
        logger.info('Starting %s phase.', phase)

        if phase == 'training':
            training(config, config['training']) # Mandatory arg

        logger.info('Finished %s phase.', phase)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for config_filename in sys.argv[1:]:
            main(config_filename)
    else:
        main('sample_config.json')


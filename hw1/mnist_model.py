import sys
import json
import logging
import predictive_model
import data_provider
from torch.autograd import Variable
import pickle


def train_epoch(model, train_data_provider, current_epoch, report_period=10):
    """
    Function that trains a model for one epoch
    :param model:
    :param train_data_provider:
    :param current_epoch:
    :param report_period:
    :return:
    """
    train_loader = train_data_provider.loader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        pred, loss = model.train_batch(data, target)
        if batch_idx % report_period == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                current_epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))


def validate_model(model, test_data_provider):
    """
    Function that validates a model
    :param model:
    :param test_data_provider:
    :return:
    """
    model.start_prediction()
    test_loader = test_data_provider.loader
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output, pred, loss_val = model.validate_batch(data, target)
        test_loss += loss_val
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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


if __name__ == "__main__":
    # initialization
    logger = logging.getLogger('main')

    # arg parse
    if len(sys.argv) < 2:
        print('Need a config filename!')
        sys.exit(1)
    config_filename = sys.argv[1]

    # parse config
    config = build_config(config_filename, logger)

    # load data
    train_provider = data_provider.DataProvider(file_dir="train_labeled.p", train=True)
    logger.info('train provider loaded')
    validation_provider = data_provider.DataProvider(file_dir="validation_data.p", train=False)
    logger.info('validation provider loaded')

    # start training
    model = predictive_model.PredictiveModel(config)
    for epoch in range(1, 11):
        model.start_train()
        train_epoch(model, train_provider, epoch)
        model.start_prediction()
        validate_model(model, validation_provider)

    # save model
    pickle.dump(model, open("sample_save_model.m", "wb"))
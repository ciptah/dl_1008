import sys
import json
import logging
import predictive_model
import data_provider
from torch.autograd import Variable
import pickle
import numpy as np
import pandas as pd
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import torch


def elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def preprocess_img(img, width, height, limits_ranslate=(-3, 3), limits_rotate=(-20, 20)):
    # add noise
    img += np.random.normal(scale=0.2, size=(width, height))

    # translation
    dx = np.random.choice(range(limits_ranslate[0], limits_ranslate[1] + 1))
    dy = np.random.choice(range(limits_ranslate[0], limits_ranslate[1] + 1))
    img[:, :] = ndimage.interpolation.shift(img, (dx, dy))

    # rotation
    degree = np.random.choice(range(limits_rotate[0], limits_rotate[1] + 1))
    ndimage.rotate(img, degree, reshape=False, output=img[:, :])

    # elastic distrotion
    img[:, :] = elastic_transform(img, 36, 8)
    # return img_arr


def default_augmentation(data):
    batch_num = data.size()[0]
    width = data.size()[2]
    height = data.size()[3]

    data = data.data.numpy()
    # output = np.ones((batch_num,1,width,height))
    for i in range(0, batch_num):
        # output[i,0,:,:] = preprocess_img(data[i,0,:,:], width, height)
        preprocess_img(data[i, 0, :, :], width, height)

        # return Variable(torch.Tensor(output))


def train_epoch(model, train_data_provider, current_epoch, report_period=100, unlablled=False,
                aug_funct=default_augmentation):
    """
    Function that trains a model for one epoch
    :param model:
    :param train_data_provider:
    :param current_epoch:
    :param report_period:
    :return:
    """
    train_loader = train_data_provider.loader
    total_num = 0
    correct_num = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if aug_funct is not None:
            aug_funct(data)
        pred_dist, loss = model.train_batch(data, target)
        if not unlablled:
            pred = pred_dist[0].data.max(1)[1]
            correct_num += (pred == target.data).sum()
            total_num += len(pred)
        if batch_idx % report_period == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                current_epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss))
    if not unlablled:
        print("Train Epoch {0}, train_acc = {1}".format(current_epoch, float(correct_num) / total_num))


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
        _, pred, loss_val = model.validate_batch(data, target)
        test_loss += loss_val
        correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return float(correct) / len(test_loader.dataset)


def predict_model(model, pred_data_loader):
    model.start_prediction()
    pred_loader = pred_data_loader.loader
    label_predict = []
    for _, content in enumerate(pred_loader):
        data = Variable(content[0])
        output = model.predict_batch(data)
        pred = output[1].numpy().tolist()
        label_predict += [x[0] for x in pred]
    df = pd.DataFrame({"ID": range(len(label_predict)), "label": label_predict})
    return df


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
    config = build_config("sample_config.json", logger)

    # load data
    # comment out loader if you don't need. they take a hell lot of time!
    train_provider = data_provider.DataProvider(file_dir="train_labeled.p", train=True)
    logger.info('train provider loaded')
    train_unlabeled = data_provider.DataProvider(file_dir="train_unlabeled.p", train=True)
    logger.info('train unlabelled provider loaded')
    validation_provider = data_provider.DataProvider(file_dir="validation_data.p", train=False)
    logger.info('validation provider loaded')
    #pred_label = data_provider.DataProvider(file_dir="test.p", train=False)
    #logger.info('test provider loaded')

    # Num of epochs
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    logger.info('running for %d epochs', num_epochs)

    # start training
    model = predictive_model.PredictiveModel(config)
    save_dir = "./data/model/"
    save_name = config.get('model', 'basic')
    try:
        best_acc = 0
        for epoch in range(1, 2000):
            model.model.current_epoch_num = epoch
            model.start_train()
            train_epoch(model, train_provider, epoch)
            if epoch > 200:
                train_epoch(model, train_unlabeled, epoch, unlablled=True)
            model.start_prediction()
            acc = validate_model(model, validation_provider)

            print("Epoch {0}, validation acc = {1}".format(epoch, acc))
            if acc > best_acc:
                pickle.dump(model, open(os.path.join(save_dir, "{0}_best.p".format(save_name)), "wb"))
                best_acc = acc
                logger.info('best model found at acc = {0}, best model saved to {1}'.format(acc, os.path.join(save_dir,
                                                                                                              "{0}_best.p".format(
                                                                                                                  save_name))))
            else:
                pickle.dump(model, open(os.path.join(save_dir, "{0}_latest.p".format(save_name)), "wb"))
                logger.info('model saved to {0}'.format(os.path.join(save_dir, "{0}_latest.p".format(save_name))))
    finally:
        pickle.dump(model, open(os.path.join(save_dir, "{0}_latest.p".format(save_name)), "wb"))
        logger.info('model saved to {0}'.format(os.path.join(save_dir, "{0}_latest.p".format(save_name))))

    # save model
    with open('sample_save_model.m', 'wb') as save_file:
        pickle.dump(model, save_file)

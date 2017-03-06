"""Main program entry point."""

import data_provider
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import predictive_model
import sys
import torch

from config import build_config
from torch.autograd import Variable

# initialization
logger = logging.getLogger('main')


def unlabeled_training(
        config,
        unlabeled_loader):
    unlabeled_model = predictive_model.get_unlabeled_model(config)
    if unlabeled_model == None:
        return

    logger.info('pre-training: has %d unlabeled examples', unlabeled_loader.example_count())
    num_unlabeled_epochs = config.get('training', {}).get('num_unlabeled_epochs', 10)

    unlabeled_model.start_train()
    logger.info('running unlabeled training for %d epochs', num_unlabeled_epochs + 1)
    for epoch in range(1, num_unlabeled_epochs + 1):
        # Single epoch, unlabeled mode
        minibatches = unlabeled_loader
        for batch_idx, (data, unused_) in enumerate(minibatches):
            data = Variable(data)
            loss = unlabeled_model.train_batch(data)
            if batch_idx % 5 == 0:
                print('Pretrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    minibatches.example_count(),
                    100. * batch_idx / len(minibatches), loss))
        unlabeled_model.epoch_done(epoch - 1)
    unlabeled_model.training_done()


def train_epoch(model,
                train_loader,
                current_epoch,
                report_period=100,
                unlabeled=False):
    """
    Function that trains a classifier model for one epoch
    :param model:
    :param train_loader:
    :param current_epoch:
    :param report_period:
    :return:
    """
    total_num = 0
    correct_num = 0
    batch_loss = []
    batch_acc = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target.long())
        if unlabeled:
            target.data.fill_(-1)
        pred_dist, loss = model.train_batch(data, target)
        if not unlabeled:
            pred = pred_dist[0].data.max(1)[1]
            correct_num += (pred == target.data).sum()
            total_num += len(pred)
        if batch_idx % report_period == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                current_epoch,                          # In epoch 6...
                batch_idx * len(data),                  # 360 examples ...
                train_loader.example_count(),           # out of 10000 ...
                100. * batch_idx / len(train_loader),   # (3.6%)
                loss))
        batch_loss.append(loss)
    if not unlabeled:
        acc = float(correct_num) / total_num
        print("Train Epoch {0}, train_acc = {1}".format(current_epoch, acc))
        batch_acc.append(acc)
    return batch_loss, batch_acc


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
        test_loss,                                   # 0.1023
        correct,                                     # 17
        test_loader.example_count(),                 # out of 100
        100. * correct / test_loader.example_count()))  # (10%)
    return float(correct) / test_loader.example_count()


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


def main():
    # arg parse
    if len(sys.argv) < 2:
        print('Need a config filename!')
        sys.exit(1)
    config_filename = sys.argv[1]

    # parse config
    config = build_config(config_filename, logger)

    # load data
    # comment out loader if you don't need. they take a hell lot of time!
    train_provider = data_provider.DataProvider(file_dir="train_labeled.p", train=True)
    logger.info('train provider loaded')
    train_unlabeled = data_provider.DataProvider(file_dir="train_unlabeled.p", train=True)
    logger.info('train unlabeled provider loaded')
    validation_provider = data_provider.DataProvider(file_dir="validation_data.p", train=False)
    logger.info('validation provider loaded')
    pred_label = data_provider.DataProvider(file_dir="test.p", train=False)
    logger.info('test provider loaded')

    # unlabeled training
    if not config.get('skip_unlabeled_training'):
        unlabeled_training(config, train_unlabeled.loader)

    if config.get('skip_labeled_training', False):
        logger.info('skipping labeled training.')
        sys.exit(0)

    # Num of epochs
    num_epochs = config.get('training', {}).get('num_epochs', 10)
    epochs_before_unlabeled = config.get('training', {}).get('epochs_before_unlabeled', 200)
    reporting_period = config.get('training', {}).get('report_every_n_batch', 100)
    logger.info('running for %d epochs', num_epochs)

    # start training
    pmodel = predictive_model.PredictiveModel(config)
    augmented_loader = predictive_model.augment_training_data(
            config, train_provider.loader)
    save_dir = "./data/model/"
    save_name = config.get('model', 'basic')

    training_loss_file = 'training_loss.csv'
    u_training_loss_file = 'u_training_loss.csv'
    training_acc_file = 'training_acc.csv'
    validation_acc_file = 'validation_acc.csv'

    training_losses = []
    u_training_losses = []
    training_acc = []
    validation_acc = []
    try:
        best_acc = 0
        for epoch in range(1, num_epochs + 1):
            pmodel.model.current_epoch_num = epoch
            pmodel.start_train()
            epoch_loss, epoch_acc = train_epoch(pmodel, augmented_loader, epoch, reporting_period)
            if epoch > epochs_before_unlabeled:
                u_epoch_loss, _ = train_epoch(pmodel, train_unlabeled.loader, epoch, reporting_period, unlabeled=True)
                u_training_losses.append(u_epoch_loss)
            training_losses.append(epoch_loss)
            training_acc.append(epoch_acc)
            pmodel.start_prediction()
            acc = validate_model(pmodel, validation_provider)
            validation_acc.append(acc)

            print("Epoch {0}, validation acc = {1}".format(epoch, acc))
            if acc > best_acc:
                pickle.dump(pmodel, open(os.path.join(save_dir, "{0}_best.p".format(save_name)), "wb"))
                best_acc = acc
                logger.info('*** NEW HIGH SCORE at acc = {0}, best model saved to {1} ***'.format(
                    acc, os.path.join(save_dir, "{0}_best.p".format(save_name))))
            else:
                pickle.dump(pmodel, open(os.path.join(save_dir, "{0}_latest.p".format(save_name)), "wb"))
                logger.info('model saved to {0}'.format(os.path.join(save_dir, "{0}_latest.p".format(save_name))))
    finally:
        pickle.dump(pmodel, open(os.path.join(save_dir, "{0}_latest.p".format(save_name)), "wb"))
        logger.info('model saved to {0}'.format(os.path.join(save_dir, "{0}_latest.p".format(save_name))))

        # Save training history to file for plotting.
        np.savetxt(training_loss_file, np.array(training_losses))
        np.savetxt(training_acc_file, np.array(training_acc))
        np.savetxt(u_training_loss_file, np.array(u_training_losses))
        np.savetxt(validation_acc_file, np.array(validation_acc))

    # save model
    with open('sample_save_model.m', 'wb') as save_file:
        pickle.dump(pmodel, save_file)


if __name__ == "__main__":
    main()

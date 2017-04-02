# RNN/LSTM language model trainer.
# Supports parameter searching (eventually)

import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import RNNModel
import logging
import config
import json
import sys
import pickle
import numpy as np
import os
import torch.optim as O

import data
import model

def run(args, config, min_test_loss):
    # Change log file
    fileh = logging.FileHandler(args.logfile, 'w')
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    fileh.setFormatter(formatter)

    logger = logging.getLogger('')  # root logger
    logger.setLevel(logging.INFO)
    # Second handler is the file logger.
    for hdlr in logger.handlers[1:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.addHandler(fileh)      # set the new handler
    logger = logging.getLogger('run')

    logger.info('CONFIGURATION: %s', json.dumps(config, indent=2))

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    init_state = torch.get_rng_state()

    logger.info('rng state: %s', init_state)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.Corpus(args.data, args.vocab_size)

    def batchify(data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        return data

    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)

    ###############################################################################
    # Build the model
    ###############################################################################
    def load_embedding(corpus, glove_file="data/glove/glove.6B.{0}d.txt", line_to_load=100000):
        """
        Function that populates a dictionary with word embedding vectors
        """
        # resolve glove file
        glove_file = glove_file.format(args.emsize)
        if not os.path.exists(glove_file):
            logger.error("glove_file {0} not exist!".format(glove_file))
            raise ValueError("glove_file {0} not exist!".format(glove_file))
        ctr = 0
        # This is the thing to return
        word_emb = np.random.uniform(-0.1, 0.1, size=(len(corpus.dictionary), args.emsize))
        found_words = 0
        with open(glove_file, "r") as f:
            for i, line in enumerate(f):
                ctr += 1
                contents = line.split()
                word = contents[0].lower()
                if word in corpus.dictionary.word2idx:
                    idx = corpus.dictionary.word2idx[word]
                    word_emb[idx,:] = np.asarray(contents[1:]).astype(float)
                    found_words += 1
                if ctr >= line_to_load:
                    break
        logger.info('found: %d', found_words)
        return torch.Tensor(word_emb)

    ntokens = len(corpus.dictionary)
    preload_emb = load_embedding(corpus) if args.initialization["word_embedding"] == "glove" else None
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers,
                     emb_init_method=args.initialization["word_embedding"],
                     weight_init_method=args.initialization["weights"],
                     preload_emb=preload_emb)
    criterion = nn.CrossEntropyLoss()
    if args.optim == 'adam':
        opt = O.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = O.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ###############################################################################
    # Training code
    ###############################################################################

    def clip_gradient(model, clip):
        """Computes a gradient clipping coefficient based on gradient norm."""
        totalnorm = 0
        for p in model.parameters():
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
        totalnorm = math.sqrt(totalnorm)
        return min(1, args.clip / (totalnorm + 1e-6))


    def repackage_hidden(h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(repackage_hidden(v) for v in h)


    def get_batch(source, i, evaluation=False):
        seq_len = min(args.sequence_length, len(source) - 1 - i)
        data = Variable(source[i:i + seq_len], volatile=evaluation)
        target = Variable(source[i + 1:i + 1 + seq_len].view(-1))
        return data, target


    def evaluate(data_source):
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        for i in range(0, data_source.size(0) - 1, args.sequence_length):
            data, targets = get_batch(data_source, i, evaluation=True)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss[0] / len(data_source)


    def train():
        total_loss = 0
        start_time = time.time()
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(args.batch_size, hidden_init_method=args.initialization["hidden_state"])
        iter_idx = range(0, train_data.size(0) - 1, args.sequence_length)
        if args.shuffle:
            np.random.shuffle(iter_idx)
        for batch, i in enumerate(iter_idx):
            data, targets = get_batch(train_data, i)
            hidden = repackage_hidden(hidden)
            model.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            clipped_lr = lr * clip_gradient(model, args.clip)
            for param_group in opt.param_groups:
                param_group['lr'] = clipped_lr
            opt.step()

            total_loss += loss.data

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                ppl = 0
                try:
                    ppl = math.exp(cur_loss)
                except OverflowError:
                    ppl = float('inf')
                logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.sequence_length, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, ppl))
                total_loss = 0
                start_time = time.time()

    # Loop over epochs.
    lr = args.lr
    prev_val_loss = None
    epoch_logs = []
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        logger.info('-' * 89)
        time_s = time.time() - epoch_start_time
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, time_s,
                                           val_loss, math.exp(val_loss)))
        logger.info('-' * 89)
        epoch_logs.append({
            'epoch': epoch,
            'time_s': time_s,
            'val_loss': val_loss,
            'val_ppl': math.exp(val_loss)
        })
        # Anneal the learning rate.
        if prev_val_loss and val_loss > prev_val_loss:
            lr /= 4.0
            logger.info('new learning rate: {}'.format(lr))
        prev_val_loss = val_loss

    # Run on test data and save the model.
    test_loss = evaluate(test_data)
    logger.info('=' * 89)
    logger.info('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logger.info('=' * 89)
    if args.save != '' and test_loss < min_test_loss:
        with open(args.save, 'wb') as f:
            torch.save(model, f)
        with open('models/best_model.pt', 'wb') as f:
            torch.save(model, f)

    # Log results in a machine-readable JSON.
    result = {}
    result['config'] = config
    result['epoch_logs'] = epoch_logs
    result['test_loss'] = test_loss
    result['test_ppl'] = math.exp(test_loss)
    with open(args.results, 'w') as r:
        json.dump(result, r, indent=2)

    # Revert random state.
    torch.set_rng_state(init_state)
    return test_loss

templatefile = sys.argv[1]
global_config = config.build_config_template(templatefile)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

min_test_loss = 1e90
for conf in config.generate_configs(global_config['template']):
    args = AttrDict(conf)
    test_loss = run(args, conf, min_test_loss)
    if test_loss < min_test_loss:
        min_test_loss = test_loss

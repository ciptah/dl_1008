# Execute model on PTB test data and print the perplexity.

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging

import data
import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

parser = argparse.ArgumentParser(description='PyTorch PTB Tester')

parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./best_model.pt',
                    help='model checkpoint to use')
parser.add_argument('--sequence_length', type=int, default=20,
                    help='how long each sentence is.')
args = parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
print 'vocab size: ', ntokens

criterion = nn.CrossEntropyLoss()

eval_batch_size = 10

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i, evaluation=False):
    seq_len = min(args.sequence_length, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

test_data = batchify(corpus.test, eval_batch_size)

def evaluate(data_source):
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.sequence_length):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        batch_loss = len(data) * criterion(output_flat, targets).data
        total_loss += batch_loss
        hidden = repackage_hidden(hidden)
        logger.info('batch loss: %.2f | total loss: %.2f', batch_loss[0], total_loss[0])
    return total_loss[0] / len(data_source)

# Run on test data and save the model.
test_loss = evaluate(test_data)
logger.info('=' * 89)
logger.info('| test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 89)

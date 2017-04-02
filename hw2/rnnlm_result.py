# Execute model on PTB test data and print the perplexity.

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    target = Variable(source[i+1:i+1+seq_len])
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
    per_word = torch.Tensor(args.sequence_length)
    pw_count = torch.Tensor(args.sequence_length)
    per_word.fill_(0.0)
    pw_count.fill_(0)
    for i in range(0, data_source.size(0) - 1, args.sequence_length):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        # Dimension 0 is the sequence length.
        batch_loss = 0.0
        for w in range(len(data)):
            loss = criterion(output[w,:,:], targets[w,:]).data[0]
            per_word[w] += loss
            pw_count[w] += 1
            batch_loss += loss
        total_loss += batch_loss
        hidden = repackage_hidden(hidden)
        logger.info('batch loss: %.2f | total loss: %.2f', batch_loss, total_loss)
    logger.info(per_word)
    logger.info(pw_count)

    losses = per_word / pw_count
    with open('per_word_loss.csv', 'w') as w:
        for x in losses.numpy():
            w.write('{}\n'.format(x))

    plt.figure()
    plt.bar(range(args.sequence_length), losses.numpy())
    plt.savefig('per_word_loss.png')
    logger.info('saved per_word_loss.png')

    plt.figure()
    plt.bar(range(args.sequence_length), np.exp(losses.numpy()))
    plt.savefig('per_word_pplx.png')
    logger.info('saved per_word_pplx.png')

    # Embed using T-SNE
    emb_np = list(model.encoder.parameters())[0].data.numpy()
    logger.info('embeddings: %s', emb_np.shape)
    tsne = TSNE()
    emb_tsne = tsne.fit_transform(emb_np)

    # Get some labels
    n_labels = 100 # How many text labels to print?
    my_labels = list(corpus.dictionary.idx2word)
    np.random.shuffle(my_labels)
    my_labels = my_labels[:n_labels]
    labels_print = [(corpus.dictionary.word2idx[x], x) for x in my_labels]

    # Print a word embedding map.
    plt.figure(figsize=(15, 15))
    plt.scatter(emb_tsne[:,0], emb_tsne[:,1], marker='.', color='cyan')
    for idx, label in labels_print:
        emb_x, emb_y = emb_tsne[idx,:]
        plt.text(emb_x, emb_y, label, fontsize=14)
    plt.savefig('word_embeddings.png')
    logger.info('saved word_embeddings.png')

    return total_loss / len(data_source)

# Run on test data and save the model.
test_loss = evaluate(test_data)
logger.info('=' * 89)
logger.info('| test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logger.info('=' * 89)

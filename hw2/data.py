import os
import torch
import logging
from collections import Counter

logger = logging.getLogger('corpus')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

unk_token = '<unk>'
eos_token = '<eos>'

class Corpus(object):
    def __init__(self, path, limit=-1):
        self.dictionary = Dictionary()

        self.init_dict(os.path.join(path, 'train.txt'), limit)

        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def init_dict(self, path, limit):
        # Add words to the dictionary
        counter = Counter()
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                counter.update(words)

        # Limit
        common = sorted(counter.most_common(), key=lambda x: (-x[1], x[0]))
        common = [x[0] for x in common]
        if limit > 0:
            common = common[:limit]
        # Necessary tokens
        if unk_token not in common:
            common.insert(0, unk_token)
            common = common[:-1]
        if eos_token not in common:
            common.insert(0, eos_token)
            common = common[:-1]

        for word in common:
            self.dictionary.add_word(word)

        logger.info('init dict from %s', path)
        logger.info('sample: %s ... %s', common[:5], common[-5:])
        if limit > 0:
            logger.info('limited to: %d', limit)

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = []
            lines = 0
            replaced_with_unk = 0
            for line in f:
                lines += 1
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.dictionary.word2idx:
                        replaced_with_unk += 1
                        word = unk_token
                    tokens.append(self.dictionary.word2idx[word])

            logger.info('parsing %s', path)
            logger.info('%d lines', lines)
            logger.info('%d words replaced with <unk>', replaced_with_unk)

            # Return a loooong 1D tensor containing all input data concatenated
            # together.
            return torch.LongTensor(tokens)

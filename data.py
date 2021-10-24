import os
from collections import Counter, OrderedDict

import torch


class Vocab(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2idx['[UNK]'] = 0
        self.idx2word.append('[UNK]')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab_size = None):
        self.vocab = Vocab()
        self.vocab_size = vocab_size
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        counter = Counter()
        with open(path, 'r', encoding = 'utf8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                counter.update(words)
        for word, count in counter.most_common(self.vocab_size):
            self.vocab.add_word(word)
        
        with open(path, 'r', encoding = 'utf8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    if word in self.vocab.word2idx.keys():
                        ids.append(self.vocab.word2idx[word])
                    else:
                        ids.append(self.vocab.word2idx['[UNK]'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data


def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i: i + seq_len]
    target = source[i + 1: i + 1 + seq_len].view(-1)
    return data, target
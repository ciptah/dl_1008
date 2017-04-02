import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 emb_init_method="random", weight_init_method="random", preload_emb=None):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        self.decoder = nn.Linear(nhid, ntoken)
        if emb_init_method == "glove":
            self.preload_emb = preload_emb
        self.init_weights(emb_init_method=emb_init_method, weight_init_method=weight_init_method)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, emb_init_method="random", weight_init_method="random"):
        initrange = 0.1
        # word embedding
        if emb_init_method == "random":
            self.encoder.weight.data.uniform_(-initrange, initrange)
        elif emb_init_method == "glove":
            self.encoder.weight.data = self.preload_emb
        # weights
        if weight_init_method == "random":
            self.decoder.weight.data.uniform_(-initrange, initrange)
        elif weight_init_method == "zero":
            self.decoder.weight.data.zero_()
        # bias
        self.decoder.bias.data.fill_(0)


    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, hidden_init_method="zero"):
        weight = next(self.parameters()).data
        if hidden_init_method == "zero":
            if self.rnn_type == 'LSTM':
                return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                        Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
            else:
                return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        elif hidden_init_method == "random":
            if self.rnn_type == 'LSTM':
                return (Variable(weight.new(self.nlayers, bsz, self.nhid).random_(-1, 2)),
                        Variable(weight.new(self.nlayers, bsz, self.nhid).random_(-1, 2)))
            else:
                return Variable(weight.new(self.nlayers, bsz, self.nhid).random_(-1, 2  ))

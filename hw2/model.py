import torch
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 emb_init_method="random", weight_init_method="random", preload_emb=None,
                 dropout=0.5):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM","GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
        else:
            nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        self.decoder = nn.Linear(nhid, ntoken)
        self.dropout_in = nn.Dropout(p=dropout)
        self.dropout_out = nn.Dropout(p=dropout)

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
        emb = self.dropout_in(emb)
        output, hidden = self.rnn(emb, hidden)
        output = self.dropout_out(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, hidden_init_method="zero"):
        weight = next(self.parameters()).data
        def rd():
            return torch.Tensor(self.nlayers, bsz, self.nhid).random_(-1, 2).cuda()
        if hidden_init_method == "zero":
            if self.rnn_type == 'LSTM':
                return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                        Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
            else:
                return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        elif hidden_init_method == "random":
            if self.rnn_type == 'LSTM':
                return (Variable(rd()), Variable(rd()))
            else:
                return Variable(rd())

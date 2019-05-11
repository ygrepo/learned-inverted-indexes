import torch.nn as nn


class ShallowNetwork(nn.Module):

    def __init__(self, hidden_size=3, bias=False, non_linearity="tanh"):
        """
        hidden_size: a list of the sizes (num of neurons) of the hidden layers
        """
        super(ShallowNetwork, self).__init__()
        non_linearities = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "leaky_relu": nn.LeakyReLU()}
        self.hidden_size = hidden_size
        self.input_dim = 1
        self.output_dim = 1
        self.linear_1 = nn.Linear(self.input_dim, self.hidden_size, bias)
        self.nl_1 = non_linearities[non_linearity]
        self.linear_2 = nn.Linear(self.hidden_size, self.output_dim, bias)
        self.nl_2 = nn.ReLU()

    def forward(self, input):
        output = self.linear_1(input)
        output = self.nl_1(output)
        output = self.linear_2(output)
        output = self.nl_2(output)
        return output


class LSTMAE(nn.Module):
    def __init__(self, mode, input_dim, latent_dim, num_layers):
        super(LSTMAE, self).__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_dim = 1

        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        self.decoder = nn.LSTM(self.latent_dim, self.output_dim, self.num_layers)

    def forward(self, input_seqs, input_lengths, hidden=None):
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, padding_value=0.0, batch_first=False)
        lengths = [i - 1 for i in input_lengths] if self.mode == "doc2doc" else input_lengths
        input_packed = nn.utils.rnn.pack_padded_sequence(input_padded, lengths, batch_first=False)
        output_packed, hidden = self.encoder(input_packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, padding_value=0.0, batch_first=False)
        y, hidden = self.decoder(output)
        return y.squeeze(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.latent_dim),
                weight.new_zeros(self.num_layers, bsz, self.latent_dim))


class LII_LSTM(nn.Module):
    def __init__(self, rnn_type, ninp, nhid, nlayers=1, dropout=0):
        super(LII_LSTM, self).__init__()
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                   options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.linear = nn.Linear(self.nhid, ninp)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, input_lengths, hidden):
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, padding_value=0.0, batch_first=False)
        input_packed = nn.utils.rnn.pack_padded_sequence(input_padded, input_lengths, batch_first=False)
        output_packed, hidden = self.rnn(input_packed, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, padding_value=0.0, batch_first=False)
        max_seq_len, batch_size, _ = output.size()
        output = output.contiguous()
        output = output.view(-1, output.shape[2])
        output = self.linear(output)
        output = output.view(max_seq_len, batch_size)
        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

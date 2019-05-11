import torch.nn as nn


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


class RNNModel(nn.Module):
    def __init__(self, mode, rnn_type, input_dim, hidden_dim, nlayers):
        super(RNNModel, self).__init__()
        self.mode = mode
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.encode_dim = 100
        self.output_dim = 1
        self.encoder = nn.Linear(self.input_dim, self.encode_dim)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(self.encode_dim, self.hidden_dim, nlayers)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[self.rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--rnn` was supplied,
                                   options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.input_dim, self.encode_dim, self.nlayers,
                              nonlinearity=nonlinearity, batch_first=False)
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)  # go from hidden dim to dim of 1
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.bias.data.zero_()
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Pad inputs
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, padding_value=0.0, batch_first=False)
        lengths = [i - 1 for i in input_lengths] if self.mode == "doc2doc" else input_lengths

        # Encode inputs
        encoded = self.encoder(input_padded)

        # Pack inputs so that they aren't shown to RNN
        input_packed = nn.utils.rnn.pack_padded_sequence(encoded, lengths, batch_first=False)

        # Run inputs through RNN
        output_packed, hidden = self.rnn(input_packed, hidden)

        # Unpack outputs
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, padding_value=0.0, batch_first=False)

        # Decode outputs
        max_seq_len, batch_size, _ = output.size()
        output = self.decoder(output)
        output = output.view(max_seq_len, batch_size)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros(self.nlayers, bsz, self.hidden_dim),
                    weight.new_zeros(self.nlayers, bsz, self.hidden_dim))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.hidden_dim)

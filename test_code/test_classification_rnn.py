import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data_utils

from src.collection import Collection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data
    # return data.to(device)


def read_data():
    # Use test data
    test_collection = Collection("test_data/test_collection")
    posting_lists = []
    for _, pl in enumerate(test_collection):
        posting_lists.append(np.unique(np.array(pl[0], dtype=np.uint8)))
    return posting_lists


def get_dataset(posting_list, bsz):
    index = np.array(range(len(posting_list)), np.uint8)
    target = posting_list
    index, target = torch.from_numpy(index).float().to(device), torch.from_numpy(target).float().to(device)
    loader = data_utils.TensorDataset(index, target)
    loader_dataset = data_utils.DataLoader(loader, batch_size=bsz, shuffle=False)
    return loader_dataset


def main():
    batch_size = 1
    seq_len = 7
    seq_len = 254
    input_size = 9
    input_size = 256
    hidden_size = 255

    example = [1, 2, 3, 4, 5, 6, 7, 8]
    example = [1, 12, 32, 41, 55, 67, 79, 81]

    example = np.arange(1, 256, 1)


    embedding = nn.Embedding(input_size, hidden_size)
    rnn = torch.nn.RNN(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity='tanh'
    )
    print('rnn', rnn)

    # input = autograd.Variable(
    #     torch.LongTensor(example[:-1]).view(seq_len, batch_size)
    # )
    #
    # target = autograd.Variable(
    #     torch.LongTensor(example[1:]).view(seq_len, batch_size)
    # )
    input =  torch.LongTensor(example[:-1]).view(seq_len, batch_size)
    target = torch.LongTensor(example[1:]).view(seq_len, batch_size)

    print(input.size(), target.size())

    parameters = [p for p in rnn.parameters()] + [p for p in embedding.parameters()]
    optimizer = torch.optim.Adam(parameters)
    criterion = nn.NLLLoss()
    epoch = 0
    while True:
        embedding_input = embedding(input)
        state = torch.zeros(1, batch_size, hidden_size)
        #state = autograd.Variable(torch.zeros(1, batch_size, hidden_size))
        #hidden = model.init_hidden(args.batch_size)
        out, _ = rnn(embedding_input, None)
        #out, state = rnn(embedding_input, state)
        out_unembedded = out.view(-1, hidden_size) @ embedding.weight.transpose(0, 1)
        _, pred = out_unembedded.max(1)
        #out_unembedded = out_unembedded.view(seq_len, batch_size, input_size)
        loss = criterion(out_unembedded, target.view(-1))
        # print('epoch={}, loss={}'.format(epoch, loss.item()))
        if epoch % 500 == 0:
            #print('out={}'.format( out.view(1, -1)))
            print('epoch={}'.format(epoch))
            print('input={}'.format(input.data.view(1, -1)))
            print('target={}'.format(target.view(1, -1)))
            print('pred={}'.format(pred.view(1, -1)))

        embedding.zero_grad()
        rnn.zero_grad()
        loss.backward()
        optimizer.step()

        #   print('out={}, state={}'.format(out.size(), state.size()))
        epoch += 1


if __name__ == "__main__":
    main()

# Import external dependencies
import sys

import numpy as np
import torch
import torch.nn as nn

# Import local dependencies
sys.path.insert(0, "../src")
from collection import Collection


def get_batch(source, source_lengths, i, bsz):
    batch_size = min(bsz, len(source_lengths))
    return source[i:i + batch_size], source_lengths[i:i + batch_size]


def get_data(source, device):
    data = [d.unsqueeze(dim=1).to(device) for d in source]
    target = [d.to(device) for d in source]
    return data, target


def list_of_tensors(data):
    tensor_list = []
    for pl in data:
        tensor_list.append(torch.tensor(pl, dtype=torch.float64))
    return tensor_list


def load_test_data(posting_length_to_use):
    # Load data
    test_collection = Collection("../test_data/test_collection")
    posting_lists = []
    for _, pl in enumerate(test_collection):
        if len(pl[0]) >= posting_length_to_use:
            posting_lists.append(np.array(pl[0], dtype=np.int))
    #posting_lists.sort(key=lambda x: np.shape(x)[0], reverse=True)

    data = list_of_tensors(posting_lists)
    return data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.set_default_tensor_type(torch.DoubleTensor)


    # Load data
    data = load_test_data(128)
    print(len(data))
    pl = data[0]
    print(pl)
    input = pl.view(pl.shape[0], 1)
    print(input.shape)

    len_pl = len(pl)
    X = torch.arange(len_pl, dtype=torch.float64).to(device)
    y = pl

    input_dim = len_pl
    output_dim = len_pl
    n_hidden = int(len_pl / 10)
    model = nn.Sequential(
        nn.Linear(input_dim, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, output_dim),
    )
    model.to(device)
    parameters = [p for p in model.parameters()]
    optimizer = torch.optim.Adam(params=parameters, lr=0.01)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True,
                                                              threshold=10e-6)
    criterion = torch.nn.L1Loss()
    epoch = 0

    while True:
        predictions = model(X)
        loss = criterion(predictions, y)
        if epoch % 500 == 0:
            print('epoch={}, loss={}'.format(epoch, loss.item()))

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step(loss)

        epoch += 1


if __name__ == "__main__":
    main()

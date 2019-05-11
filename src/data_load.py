from random import shuffle

import numpy as np
import torch
import torch.utils.data as torch_data

# Import local dependencies
from collection import Collection


class DatasetPostingList(torch_data.Dataset):
    def __init__(self, data_list, pos_list):
        self.data = data_list
        self.pos_list = pos_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.pos_list[index]


class DatasetDataList(torch_data.Dataset):
    def __init__(self, data_lists, datalists_lengths):
        assert len(data_lists) == len(datalists_lengths)
        self.data_lists = data_lists
        self.datalists_lengths = datalists_lengths

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        return self.data_lists[index], self.datalists_lengths[index]


def get_canonical_dataset(logger):
    index = np.arange(0, 255, 1)
    bsz = len(index)
    target = np.arange(0, 255, 1)
    index, target = torch.from_numpy(index).float().to(device), torch.from_numpy(target).float().to(device)
    logger.info('input.size={}, target.size={}'.format(index.size(), target.size()))
    loader = torch_data.TensorDataset(index, target)
    data_loader = torch_data.DataLoader(loader, batch_size=bsz, shuffle=False)
    return data_loader


def list_of_tensors(data):
    tensor_list = []
    for pl in data:
        tensor_list.append(torch.tensor(pl, dtype=torch.float64))
    return tensor_list


def get_batch(source, source_lengths, i, bsz):
    batch_size = min(bsz, len(source_lengths))
    return source[i:i + batch_size], source_lengths[i:i + batch_size]


def get_data(device, source):
    data = [d.unsqueeze(dim=1).to(device) for d in source]
    target = [d.to(device) for d in source]
    return data, target


def load_test_data(collection_file, posting_length_to_use, shuffling=False, sampling_size_ratio=None):
    # Load data
    test_collection = Collection(collection_file)
    posting_lists = []
    for _, pl in enumerate(test_collection):
        if len(pl[0]) < posting_length_to_use:
            posting_lists.append(np.array(pl[0], dtype=np.float64))

    if shuffling:
        shuffle(posting_lists)

    if sampling_size_ratio:
        sampling_size = int(len(posting_lists) * sampling_size_ratio)
        posting_lists = posting_lists[0:sampling_size]

    posting_lists.sort(key=lambda x: np.shape(x)[0], reverse=True)

    lengths = [len(pl) for pl in posting_lists]
    data = list_of_tensors(posting_lists)

    return data, lengths

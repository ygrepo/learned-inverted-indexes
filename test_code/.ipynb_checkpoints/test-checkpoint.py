import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as data_utils
import numpy as np



def get_dataset(bsz):
    index  = np.arange(0,255,1)
    bsz = len(index)
    target  = np.arange(0,255,1)
    index, target = torch.from_numpy(index).float(), torch.from_numpy(target).float()
    #.view(-1,1,1).to(device), torch.from_numpy(target).float().view(-1,1,1).to(device)
    dataset = data_utils.TensorDataset(index, target)
    data_loader = data_utils.DataLoader(dataset, batch_size=bsz, shuffle=False)
    return data_loader

data_loader = get_dataset(1)
print(len(data_loader.dataset))
index, target = next(iter(data_loader))
index = index.view(-1,1,1)
print(index.shape)
index, target = next(iter(data_loader))
index = index.view(-1,1,1)
print(index.shape)

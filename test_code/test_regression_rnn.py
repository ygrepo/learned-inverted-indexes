import numpy as np
import torch

# Import local dependencies
import sys
sys.path.insert(0, "src")
from collection import Collection

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor.mul_(self.std).add_(self.mean)
        return tensor

class UnNormalize2(object):
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.diff = (max - min)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor.mul_(self.diff).add_(self.min)
        return tensor

def normalize(x):
    min = x.min(0, keepdim=True)[0]
    max = x.max(0, keepdim=True)[0]
    x_normed = (x - min) / (max - min)
    return x_normed

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
        tensor_list.append(torch.tensor(pl, dtype=torch.float32))
    return tensor_list


def load_test_data(posting_length_to_use):
    # Load data
    test_collection = Collection("../test_data/test_collection")
    posting_lists = []
    for _, pl in enumerate(test_collection):
        if len(pl[0]) >= posting_length_to_use:
            posting_lists.append(np.array(pl[0], dtype=np.int32))
    posting_lists.sort(key=lambda x: np.shape(x)[0], reverse=True)

    data = list_of_tensors(posting_lists)
    return data


def isclose(a, b, rel_tol=1e-09, abs_tol=0.000001):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compute_accuracy(predictions, target):
    t1 = predictions.type(torch.LongTensor).numpy()
    t2 = target.type(torch.LongTensor).numpy()
    print('target={}'.format(target.view(1, -1)))
    print('predictions={}'.format(predictions.view(1, -1)))
    print('t1={}'.format(t1.reshape(1,-1)))
    print('t2={}'.format(t2.reshape(1,-1)))
    return np.sum(t1 == t2) / np.size(t2) * 100.

def compute_accuracy2(predictions, target):
    n1 = predictions.detach().numpy()
    n2 = target.detach().numpy()
    return np.sum(n1 == n2) / np.size(n2) * 100.

def get_canonical_data(seq_len, batch_size, input_size):
    index = np.arange(0, 257, 1)
    # term 0
    index[256] = 0
    target = np.arange(0, 257, 1)
    # term 0
    target[256] = 0
    input = torch.FloatTensor(index[:-1]).view(seq_len, batch_size, input_size)
    target = torch.FloatTensor(target[:-1]).view(seq_len, batch_size, input_size)
    print('input.size={}, target.size={}'.format(input.size(), target.size()))
    return input, target

def main():
    batch_size = 1
    seq_len = 256
    input_size = 1
    hidden_size = 12


    # Load data
    data = load_test_data(128)
    print(len(data))
    pl = data[0]
    print(pl)
    input = pl.view(len(pl), batch_size, 1)

    #input, target = get_canonical_data(seq_len, batch_size, input_size)
    #input = input.view(len(input), batch_size, 1)
    print(input.mean(0), input.std(0))
    # un = UnNormalize(input.mean(0).item(), input.std(0).item())
    # input2 = (input - input.mean(0)) / input.std(0)
    # min = input.min(0, keepdim=True)[0]
    # max = input.max(0, keepdim=True)[0]
    # un2 = UnNormalize2(min, max)
    # input2 = normalize(input)
    #print(compute_accuracy2(un(input2), input))
    #print(input2)
    # print("{}".format(un(input2).view(1, -1)))
    # print(compute_accuracy(un(input2), input2))


    rnn = torch.nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        nonlinearity='tanh'
    )
    print('rnn', rnn)

    linear = torch.nn.Linear(
        in_features=hidden_size,
        out_features=1
    )
    print('linear', linear)

    parameters = [p for p in rnn.parameters()] + [p for p in linear.parameters()]
    optimizer = torch.optim.Adam(params=parameters, lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, threshold=10e-6)

    # optimizer = torch.optim.Adam(parameters)
    # optimizer = torch.optim.SGD(parameters, 20,
    #                             momentum=0,
    #                             weight_decay=0)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                           patience=0, verbose=True, factor=0.5)

    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss(reduction='mean')
    epoch = 0
    best_loss = float("inf")
    loss, prev_loss = 0., 0.
    plateau_cnt = 0
    max_plateau = 10
    predictions = None
    try:
        while True:
            out, hidden = rnn(input, None)
            predictions = linear(out)
            prev_loss = loss
            loss = criterion(predictions, input)
            if epoch % 500 == 0:
                print('epoch={}, loss={}'.format(epoch, loss.item()))

            rnn.zero_grad()
            linear.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < best_loss:
                best_loss = loss
                # print(f"Best loss: {loss:.3f}")
                # with open("checkpoint.pth.tar.best", 'wb') as f:
                #     torch.save(model, f)

            if isclose(loss, prev_loss):
                plateau_cnt += 1
                print(f"Loss:{loss:.3f} close to previous loss:{prev_loss:.3f}")
                if plateau_cnt == max_plateau:
                    accuracy = compute_accuracy(predictions, input)
                    #accuracy = compute_accuracy2(un(predictions), input)
                    print(f'Epoch: {epoch}, best loss: {loss:.3f}, accuracy {accuracy:.1f}%')
                    break
            else:
                plateau_cnt = 0

            if lr_scheduler:
                lr_scheduler.step(loss)

            epoch += 1
    except KeyboardInterrupt:
       print('Exiting from training early')


    accuracy = compute_accuracy(predictions, input)

if __name__ == "__main__":
    main()

import argparse
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


from app_config import config_logger
from data_load import load_test_data, get_batch, get_data
from model import LII_LSTM, LSTMAE



def isclose(a, b, rel_tol=1e-09, abs_tol=0.001):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compute_accuracy(predictions, target):
    t1 = predictions.type(torch.LongTensor).numpy()
    t2 = target.type(torch.LongTensor).numpy()
    # print('target={}'.format(target.view(1, -1)))
    # print('predictions={}'.format(predictions.view(1, -1)))
    return np.sum(t1 == t2) / np.size(t2) * 100.


def load_data(device, logger):
    posting_lists = load_test_data(logger, 128)
    logger.info(len(posting_lists))
    lengths = [len(pl) for pl in posting_lists]
    batch, batch_lengths = get_batch(posting_lists[:2], lengths[:2], 0, 1)
    data, target = get_data(device, batch)
    return data, lengths, target


def train(device, model, optimizer, criterion, posting_lists, lengths,
          scheduler=None, compression_scheduler=None,
          epochs=20000, batch_size=1, log_interval=50, logger=None):
    model.train()

    start_time = time.time()
    # Loop for number of epochs:
    for epoch in range(1, epochs + 1):

        # Loop for batches within data:
        for batch_idx, i in enumerate(range(0, len(lengths), batch_size)):

            batch, batch_lengths = get_batch(posting_lists, lengths, i, batch_size)
            hidden = model.init_hidden(min(batch_size, len(batch)))

            # Get data
            data, target = get_data(device, batch)

            # Zero out the grad
            optimizer.zero_grad()
            #
            # # Get output
            predictions = model(data, batch_lengths, hidden)

            # # Calculate loss
 #           target = nn.utils.rnn.pad_sequence(target, padding_value=0.0, batch_first=False)
            loss = criterion(predictions, target)

            # # Take gradient step
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            # Take scheduler step
            if scheduler:
                scheduler.step(loss)


        if epoch % log_interval == 0:
            logger.info('-' * 89)
            logger.info('| End of epoch: {:3d} | time: {:5.2f}s | loss {:5.3f} '
                        .format(epoch, (time.time() - start_time), loss))
            logger.info('-' * 89)
            start_time = time.time()


def main():
    parser = argparse.ArgumentParser(description='PyTorch LII_LSTM')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--input_size', type=int, default=1,
                        help='input size')
    parser.add_argument('--nhid', type=int, default=10,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='./models/checkpoint.pth.tar',
                        help='path to save the final model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.resume:
        with open(args.resume, 'rb') as f:
            model = torch.load(f).to(device)
            # after load the rnn params are not a continuous chunk of memory
            # this makes them a continuous chunk, and will speed up forward pass
            model.rnn.flatten_parameters()
    else:
        #model = LII_LSTM(args.model, args.input_size, args.nhid, args.nlayers, args.dropout).to(device)
        model = LSTMAE("doc2doc", args.input_size, args.nhid, args.nlayers)

    logger = config_logger('config/logging.conf', experiment_name=None, output_dir='logs')
    logger.info("Using device:{}".format(device))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    data, lengths = load_test_data("test_data/test_collection", 128)
    data = data[:1]
    lengths = lengths[:1]

    train(device, model, optimizer, criterion, data, lengths,
          scheduler=None, compression_scheduler=None,
          epochs=args.epochs, batch_size=args.batch_size, log_interval=args.log_interval, logger=logger)


if __name__ == "__main__":
    main()

import argparse
import os
import time
from collections import OrderedDict

import distiller
import distiller.apputils as apputils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from distiller.data_loggers import TensorBoardLogger, PythonLogger
from torch.utils import data as data_utils

from collection import Collection
from model import LII_LSTM

parser = argparse.ArgumentParser(description='PyTorch LII_LSTM')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--input_size', type=int, default=1,
                    help='input size')
parser.add_argument('--nhid', type=int, default=12,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='./models/checkpoint.pth.tar',
                    help='path to save the final model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

# Distiller-related arguments
SUMMARY_CHOICES = ['sparsity', 'model', 'modules', 'png', 'percentile']
parser.add_argument('--summary', type=str, choices=SUMMARY_CHOICES,
                    help='print a summary of the model, and exit - options: ' +
                         ' | '.join(SUMMARY_CHOICES))
parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                    help='configuration file for pruning the model (default is to use hard-coded schedule)')
parser.add_argument('--momentum', default=0., type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0., type=float,
                    metavar='W', help='weight decay (default: 0)')

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
    model = LII_LSTM(args.model, args.input_size, args.nhid, args.nlayers, args.dropout).to(device)

# Distiller loggers
msglogger = apputils.config_pylogger('config/logging.conf', experiment_name=None, output_dir='logs')
tflogger = TensorBoardLogger(msglogger.logdir)
tflogger.log_gradients = True
pylogger = PythonLogger(msglogger)

def export_onnx(path, batch_size, seq_len):
    msglogger.info('The model is also exported in ONNX format at {}'.
                   format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def draw_lang_model_to_file(model, png_fname):
    """Draw a language model graph to a PNG file.

    Caveat: the PNG that is produced has some problems, which we suspect are due to
    PyTorch issues related to RNN ONNX export.
    """
    try:
        # if dataset == 'wikitext2':
        batch_size = 1
        seq_len = 255
        dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size, 1).to(device)
        hidden = model.init_hidden(batch_size)
        dummy_input = (dummy_input, hidden)
        # else:
        #     msglogger.info("Unsupported dataset (%s) - aborting draw operation" % dataset)
        #     return
        g = distiller.SummaryGraph(model, dummy_input)
        distiller.draw_model_to_file(g, png_fname)
        msglogger.info("Network PNG image generation completed")

    except FileNotFoundError as e:
        msglogger.info("An error has occured while generating the network PNG image.")
        msglogger.info("Please check that you have graphviz installed.")
        msglogger.info("\t$ sudo apt-get install graphviz")
        raise e


def isclose(a, b, rel_tol=1e-09, abs_tol=0.001):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compute_accuracy(predictions, target):
    t1 = predictions.type(torch.LongTensor).numpy()
    t2 = target.type(torch.LongTensor).numpy()
    # print('target={}'.format(target.view(1, -1)))
    # print('predictions={}'.format(predictions.view(1, -1)))
    return np.sum(t1 == t2) / np.size(t2) * 100.


# def get_dataset(posting_list, bsz):
#     index = np.array(range(len(posting_list)), np.uint8)
#     target = posting_list
#     index, target = torch.from_numpy(index).float().to(device), torch.from_numpy(target).float().to(device)
#     loader = data_utils.TensorDataset(index, target)
#     loader_dataset = data_utils.DataLodataader(loader, batch_size=bsz, shuffle=False)
#     return loader_dataset


def get_dataset():
    index = np.arange(0, 255, 1)
    bsz = len(index)
    target = np.arange(0, 255, 1)
    index, target = torch.from_numpy(index).float().to(device), torch.from_numpy(target).float().to(device)
    msglogger.info('input.size={}, target.size={}'.format(index.size(), target.size()))
    loader = data_utils.TensorDataset(index, target)
    data_loader = data_utils.DataLoader(loader, batch_size=bsz, shuffle=False)
    return data_loader


def read_data():
    # Use test data
    test_collection = Collection("test_data/test_collection")
    posting_lists = []
    for _, pl in enumerate(test_collection):
        posting_lists.append(np.unique(np.array(pl[0], dtype=np.uint8)))
    return posting_lists


###############################################################################
# Training src
###############################################################################
train_dataloader = get_dataset()
val_dataloader = get_dataset()


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    hidden = model.init_hidden(args.batch_size)
    with torch.no_grad():
        index, target = next(iter(train_dataloader))
        index = index.view(-1, 1, 1)
        target = target.view(-1, 1, 1)
        predictions, hidden = model(index, hidden)
        accuracy = compute_accuracy(predictions, target)
        loss = criterion(predictions, target)
        repackage_hidden(hidden)
    return loss, accuracy


def train(optimizer, criterion, compression_scheduler=None):
    # Turn on training mode which enables dropout.
    model.train()

    batch_id = 1
    steps_per_epoch = 1
    hidden = model.init_hidden(args.batch_size)
    index, target = next(iter(train_dataloader))
    index = index.view(-1, 1, 1)
    target = target.view(-1, 1, 1)

    hidden = repackage_hidden(hidden)

    if compression_scheduler:
        compression_scheduler.on_minibatch_begin(epoch,
                                                 minibatch_id=batch_id, minibatches_per_epoch=steps_per_epoch)

    predictions, hidden = model(index, hidden)
    loss = criterion(predictions, target)

    if compression_scheduler:
        # Before running the backward phase, we allow the scheduler to modify the loss
        # (e.g. add regularization loss)
        loss = compression_scheduler.before_backward_pass(epoch, minibatch_id=batch_id,
                                                          minibatches_per_epoch=steps_per_epoch, loss=loss,
                                                          return_loss_components=False)


    optimizer.zero_grad()
    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()


    if compression_scheduler:
        compression_scheduler.on_minibatch_end(epoch, minibatch_id=batch_id,
                                               minibatches_per_epoch=steps_per_epoch)

if args.summary:
    which_summary = args.summary
    if which_summary == 'png':
        draw_lang_model_to_file(model, 'rnn.png')
    elif which_summary == 'percentile':
        percentile = 0.9
        for name, param in model.state_dict().items():
            if param.dim() < 2:
                # Skip biases
                continue
            bottomk, _ = torch.topk(param.abs().view(-1), int(percentile * param.numel()),
                                    largest=False, sorted=True)
            threshold = bottomk.data[-1]
            msglogger.info("parameter %s: q = %.2f" % (name, threshold))
    else:
        distiller.model_summary(model, which_summary)
    exit(0)

# posting_lists = read_data()
# posting_list = posting_lists[0]
# print(posting_list)

compression_scheduler = None
# if args.compress:
    # Create a CompressionScheduler and configure it from a YAML schedule file
    # compression_scheduler = distiller.config.file_config(model, None, args.compress)

# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                           patience=0, verbose=True, factor=0.5)

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                           patience=0, verbose=True, factor=0.5)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, threshold=10e-6)

criterion = torch.nn.MSELoss()

if args.compress:
#     # Create a CompressionScheduler and configure it from a YAML schedule file
    compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler, None)

try:
    best_loss = float("inf")
    best_accuracy = 0.
    val_loss, prev_loss = 0., 0.
    plateau_cnt = 0
    max_plateau = 20
    steps_completed = 1
    steps_per_epoch = 1
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):

        epoch_start_time = time.time()

        train(optimizer, criterion, compression_scheduler)
        prev_loss = val_loss
        val_loss, accuracy = evaluate()
        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', val_loss)]))

        if epoch % args.log_interval == 0:
            msglogger.info('-' * 89)
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            msglogger.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.3f} '
                           .format(epoch, (time.time() - start_time), val_loss))
            msglogger.info('-' * 89)
            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])

            tflogger.log_training_progress(stats, epoch, 0, total=1, freq=1)

            msglogger.info(f'Epoch: {epoch}, best loss: {best_loss:.3f}, accuracy {best_accuracy:.1f}%')
            start_time = time.time()

            distiller.log_training_progress(stats, model.named_parameters(), epoch, steps_completed,
                                            steps_per_epoch, args.log_interval, [tflogger])

            distiller.log_weights_sparsity(model, epoch, loggers=[tflogger, pylogger])

        with open(args.save, 'wb') as f:
            torch.save(model, f)

        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = accuracy
            with open(args.save + ".best", 'wb') as f:
                torch.save(model, f)

        if isclose(val_loss, prev_loss):
            plateau_cnt += 1
            msglogger.info(f"Loss:{val_loss:.3f} close to previous loss:{prev_loss:.3f}")
            if plateau_cnt == max_plateau:
                msglogger.info(f'Epoch: {epoch}, loss: {val_loss:.3f}, accuracy:{accuracy:.1f}% exiting...')
                break
        else:
            plateau_cnt = 0

        if lr_scheduler:
            lr_scheduler.step(val_loss)

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch)


except KeyboardInterrupt:
    msglogger.info('-' * 89)
    msglogger.info('Exiting from training early')

# Load the last saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss, accuracy = evaluate()
msglogger.info('=' * 89)
msglogger.info(f'| End of training | test loss {test_loss:5.2f}, accuracy:{accuracy:.1f}% exiting...')
msglogger.info('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

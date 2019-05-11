import argparse
# import pdb
import os
import time
from itertools import accumulate

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as torch_data

from app_config import config_logger
from data_load import load_test_data, DatasetPostingList, DatasetDataList
from model import ShallowNetwork
from quantization import compress, decompress, log_model

NUM_MODELS = 0


def plot_errors(y_pred, y):
    plt.plot(np.arange(y_pred.size(0)), y.cpu().detach().numpy(), label="Ground truth")
    plt.plot(np.arange(y_pred.size(0)), y_pred.cpu().detach().numpy(), label="Predictions")
    plt.xlabel("Posting list Index")
    plt.ylabel("Doc id")
    plt.title("Predicted vs. Ground truth Doc IDs")
    plt.legend(loc="best")
    plt.show()


def zigzag_encode(i):
    return (i >> 31) ^ (i << 1)


def save_weights_and_deltas(logger, bias, non_linearity, device, batch, model, scrap):
    # Compress model weights
    if scrap:
        compressed_weights = np.zeros(16)
        logger.info("No weights")
    else:
        compressed_weights = compress(model)

    # Predict using compressed weights
    #log_model(model)
    model_lossy_compression = decompress(compressed_weights, bias=bias,
                                         non_linearity=non_linearity, scrap=scrap).to(device)
    #log_model(model_lossy_compression)
    model_lossy_compression.eval()
    predictions = model_lossy_compression(batch[0])

    # Calculate and save deltas
    if scrap:
        deltas = batch[1]
    else:
        deltas = predictions.round() - batch[1]
    deltas_list = deltas.squeeze().tolist()
    if not isinstance(deltas_list, list):
        deltas_list = [deltas_list]
    return deltas_list


def train_posting_lists(device, delta_freqs_file, delta_docs_file, deltas_path, logger, posting_lists_dataset,
                        non_linearity="tanh", bias=False, lr=0.01, batch_size=32,
                        stop_threshold=1e-3, scrap_threshold=100,
                        epochs=20000, log_interval=1000):
    global NUM_MODELS
    posting_lists_loader = torch_data.DataLoader(
        posting_lists_dataset, batch_size=1, shuffle=False
    )
    all_deltas = []
    for posting_list, pl_length in posting_lists_loader:
        indexes = torch.arange(pl_length.item(), dtype=torch.double).to(device)
        pl_ds = DatasetPostingList(posting_list.squeeze().tolist(), indexes)
        pl_loader = torch_data.DataLoader(
            pl_ds, batch_size=batch_size, shuffle=False
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True, threshold=10e-6)
        all_deltas += train(device=device, delta_freqs_file=delta_freqs_file, delta_docs_file=delta_docs_file,
                            bias=bias, non_linearity=non_linearity, lr=lr, pl_loader=pl_loader,
                            stop_threshold=stop_threshold, scrap_threshold=scrap_threshold,
                            epochs=epochs, log_interval=log_interval, logger=logger)

    logger.info("Total number of models: {}".format(NUM_MODELS))
    all_deltas_array = np.array(all_deltas)
    average_delta = all_deltas_array.mean()
    correct_docs = (all_deltas_array == 0).sum()
    logger.info("Total number of doc ids: {}".format(len(all_deltas)))
    logger.info("Average delta: {}".format(average_delta))
    logger.info("Number of correct doc ids: {}".format(correct_docs))

    # Metrics:
    plt.hist(all_deltas, range=(-100, 100))
    plt.title("Histogram of Deltas")
    plt.xlabel("Deltas")
    deltas_hist_file = os.path.join(deltas_path, "b{}_e{}.png".format(batch_size, epochs))
    plt.savefig(deltas_hist_file, dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

def train(device, delta_freqs_file, delta_docs_file, bias, lr, non_linearity, pl_loader, scheduler=None,
          stop_threshold=1e-3, scrap_threshold=100,
          epochs=20000, log_interval=1000, logger=None):
    global NUM_MODELS

    start_time = time.time()
    deltas_pl = []

    # Loop for batches within data:
    for data, indexes in pl_loader:

        model = ShallowNetwork(3, bias=bias, non_linearity=non_linearity).to(device)
        lambda_l2 = 1e-5
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
        criterion = torch.nn.MSELoss()
        model.train()

        NUM_MODELS += 1
        data -= data[0]
        data = data.to(device).view(-1, 1)
        indexes = indexes.to(device).view(-1, 1)

        # Loop for number of epochs:
        for epoch in range(1, epochs + 1):

            # Get output
            predictions = model(indexes)

            # Calculate loss
            loss = criterion(predictions, data)
            if loss.item() < stop_threshold:
                logger.info("Loss:{:5.3f}".format(loss))
                # plot_errors(predictions, data)
                deltas_batch = save_weights_and_deltas(logger, bias, non_linearity, device,
                                                       (indexes, data), model, scrap=False)
                deltas_pl += deltas_batch
                break

            # Zero out the grad
            optimizer.zero_grad()

            # Take gradient step
            loss.backward()
            optimizer.step()

            # Take scheduler step
            if scheduler:
                scheduler.step(loss)

            if epoch % log_interval == 0:
                logger.info('-' * 89)
                logger.info('| End of epoch: {:3d} | time: {:5.2f}s | loss {:5.3f} '
                            .format(epoch, (time.time() - start_time), loss.item()))
                logger.info('-' * 89)
                start_time = time.time()

        deltas_batch = save_weights_and_deltas(logger, bias, non_linearity, device, (indexes, data), model,
                                               scrap=loss.item() > scrap_threshold)
        deltas_pl += deltas_batch

    deltas_zigzagged = [zigzag_encode(int(i)) + 1 for i in deltas_pl]
    deltas_accumulated = list(accumulate(deltas_zigzagged))
    deltas_freqs = [len(deltas_zigzagged)] + deltas_zigzagged
    deltas_docs = [len(deltas_zigzagged)] + deltas_accumulated
    deltas_freqs_array = np.array(deltas_freqs, dtype=np.uint32).ravel()
    deltas_freqs_array.tofile(delta_freqs_file)
    deltas_docs_array = np.array(deltas_docs, dtype=np.uint32).ravel()
    deltas_docs_array.tofile(delta_docs_file)
    return deltas_pl


def read_args():
    parser = argparse.ArgumentParser(description='PyTorch FF_NN')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--posting_list_size', type=int, default=128, metavar='N',
                        help='Posting list size threshold, filtering posting list with size less than this threshold')
    parser.add_argument("--loss_stop_threshold", type=float, default=1e-3,
                        help="Loss threshold for stopping training for a given model")
    parser.add_argument("--loss_scrap_threshold", type=float, default=100,
                        help="Loss threshold for determining whether to throw away a model and only save the doc ids")
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    args = parser.parse_args()
    return args.lr, args.epochs, args.posting_list_size, args.batch_size, args.loss_stop_threshold, \
           args.loss_scrap_threshold, args.seed, args.log_interval


def main():
    # Configure logger
    LOGGER = config_logger('config/logging.conf')
    BASE_DIR = os.path.abspath("")
    LOGGER.info(f"Current directory={BASE_DIR}")

    # Set device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device:{DEVICE}")
    torch.set_default_tensor_type(torch.DoubleTensor)

    # Set the random seed manually for reproducibility.
    LR, EPOCHS, POSTING_LIST_SIZE, BATCH_SIZE, STOP_THRESHOLD, SCRAP_THRESHOLD, SEED, LOG_INTERVAL = read_args()
    LOGGER.info(f"LR:{LR:5.3f} | EPOCHS:{EPOCHS:5d} | POSTING_LIST_SIZE:{POSTING_LIST_SIZE:5d} | BATCH_SIZE:{BATCH_SIZE:3d} |\
STOP_THRESHOLD:{STOP_THRESHOLD:5.3f} | SCRAP_THRESHOLD:{SCRAP_THRESHOLD:5.3f} | \
SEED:{SEED:5d} | LOG_INTERVAL:{LOG_INTERVAL:5d}")
    torch.manual_seed(SEED)
    cudnn.deterministic = True
    cudnn.benchmark = False
    BIAS = True
    NON_LINEARITY = "leaky_relu"

    # Open files for saving weights and deltas
    DELTA_DIR = os.path.join(BASE_DIR, "deltas")
    delta_freqs_file = os.path.join(DELTA_DIR, "{}_{}lr{}_b{}_e{}_deltas.freqs".format(NON_LINEARITY,
                                                                                       "bias_" if BIAS else "",
                                                                                       LR, BATCH_SIZE, EPOCHS))
    delta_docs_file = os.path.join(DELTA_DIR, "{}_{}lr{}_b{}_e{}_deltas.docs".format(NON_LINEARITY,
                                                                                     "bias_" if BIAS else "",
                                                                                     LR, BATCH_SIZE, EPOCHS))
    LOGGER.info("Saving deltas to: {} and {}".format(delta_freqs_file, delta_docs_file))
    delta_freqs_binfile = open(delta_freqs_file, "wb")
    delta_docs_binfile = open(delta_docs_file, "wb")
    deltas_dummy = np.array([1, 1], dtype=np.uint32).ravel()
    deltas_dummy.tofile(delta_docs_binfile)

    # Load data
    TEST_FILE = "test_data/test_collection"
    posting_lists, pl_lengths = load_test_data(TEST_FILE, POSTING_LIST_SIZE, shuffling=False, sampling_size_ratio=0)
    pl_ds = DatasetDataList(posting_lists, pl_lengths)
    LOGGER.info("Loaded {} posting lists".format(len(pl_ds)))

    # Run training
    train_posting_lists(device=DEVICE, delta_freqs_file=delta_freqs_binfile, delta_docs_file=delta_docs_binfile,
                        deltas_path=DELTA_DIR,
                        logger=LOGGER, posting_lists_dataset=pl_ds,
                        bias=BIAS, non_linearity=NON_LINEARITY,
                        lr=LR, batch_size=BATCH_SIZE, stop_threshold=STOP_THRESHOLD,
                        scrap_threshold=SCRAP_THRESHOLD, epochs=EPOCHS, log_interval=LOG_INTERVAL)

    # Close files
    delta_freqs_binfile.close()
    delta_docs_binfile.close()


if __name__ == "__main__":
    main()

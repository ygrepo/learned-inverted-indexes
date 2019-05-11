# Import external dependencies
import argparse
import logging
import os
from itertools import accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Import local dependencies
from collection import Collection
from global_model import RNNModel


def embed(input_seqs, term_id):
    embedding = input_seqs.unsqueeze(dim=1).repeat(1, 2)
    embedding.index_fill_(1, torch.tensor([1]), term_id)
    return embedding


def list_of_tensors(source, mode="doc2doc"):
    data = []
    labels = []
    for pl in source:
        if mode == "doc2doc":
            data.append(embed(torch.tensor(pl[0], dtype=torch.float32), pl[1]))
        else:
            data.append(embed(torch.arange(pl[0].size, dtype=torch.float32), pl[1]))
        labels.append(torch.tensor(pl[0], dtype=torch.float32))
    return data, labels


def get_batch(data, target, source_lengths, i, bsz):
    batch_size = min(bsz, len(source_lengths))
    return data[i:i+batch_size], target[i:i+batch_size], source_lengths[i:i+batch_size]


def get_data(device, data, target, mode="doc2doc"):
    if mode == "doc2doc":
        data = [d[:-1].to(device) for d in data]
        target = [t[1:, 0].to(device) for t in target]
    else:
        data = [d.to(device) for d in data]
        target = [t.to(device) for t in target]
    return data, target


def train(device, model, optimizer, source_data, source_labels, lengths,
          mode="doc2doc", scheduler=None, epochs=2000, batch_size=3, log_interval=10):
    model.train()
    epoch_losses = []
    # Loop for number of epochs:
    for e in range(1, epochs + 1):
        epoch_loss = 0

        # Loop for batches within data:
        for batch_idx, i in enumerate(range(0, len(lengths), batch_size)):
            batch_data, batch_target, batch_lengths = get_batch(source_data, source_labels, lengths, i, batch_size)
            hidden = model.init_hidden(min(batch_size, len(batch_data)))

            # Get data
            data, target = get_data(device, batch_data, batch_target, mode)

            # Zero out the grad
            optimizer.zero_grad()

            # Get output
            prediction, _ = model(data, batch_lengths, hidden)

            # Calculate loss
            target = nn.utils.rnn.pad_sequence(target, padding_value=0.0, batch_first=False)
            loss = F.mse_loss(prediction, target)
            epoch_loss += loss.item()

            # Take gradient step
            loss.backward()
            optimizer.step()

            # Take scheduler step
            if scheduler:
                scheduler.step(loss)
        epoch_losses.append(epoch_loss)

        # Print loss and plot predictions vs. ground truth
        if e % min(log_interval, epochs) == 0:
            logging.info("Train Epoch {}: Loss - {}, Avg Loss - {}".format(e,
                                                                           epoch_losses[-1],
                                                                           sum(epoch_losses) / (e + 1)))


def evaluate_doc2doc(device, model, posting_lists, lengths, primer_tokens=1):
    model.eval()
    predictions = []
    for idx, pl in enumerate(posting_lists):
        term_id = pl[0, 1]
        hidden = model.init_hidden(1)
        # Get first output
        prediction, hidden = model([pl[:primer_tokens].to(device)], [primer_tokens+1], hidden)
        next_prediction = prediction
        for i in range(lengths[idx] - primer_tokens):
            next_token = next_prediction.repeat(1, 2)
            next_token[0, 1] = term_id
            next_prediction, hidden = model([next_token.to(device)], [2], hidden)
            prediction = torch.cat((prediction, next_prediction))
        predictions.append(prediction.squeeze())
    return predictions


def evaluate_index2doc(device, model, indexes, lengths):
    model.eval()
    predictions = []
    for i, pl in enumerate(indexes):
        hidden = model.init_hidden(1)
        predictions.append(model([pl.to(device)], [lengths[i]], hidden)[0].squeeze())
    return predictions


def zigzag_encode(i):
    return (i >> 31) ^ (i << 1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DL SP19 Project: LII Global Model")
    parser.add_argument("--mode", type=str, default="doc2doc", metavar="M",
                        help="Predict doc2doc (default) or index2doc")
    parser.add_argument("--rnn", type=str, default="GRU", metavar="R",
                        help="Type of RNN to be used: LSTM vs GRU (default: GRU)")
    parser.add_argument("--hidden", type=int, default=10, metavar="H",
                        help="Number of hidden units in RNN (default: 10)")
    parser.add_argument("--layers", type=int, default=1, metavar="L",
                        help="Number of layers in RNN (default: 10)")
    parser.add_argument("--epochs", type=int, default=1000, metavar="E",
                        help="Number of epochs for training (default: 1000)")
    parser.add_argument("--batch", type=int, default=247, metavar="B",
                        help="Batch size used during training (default: 247)")
    parser.add_argument("--log", type=int, default=10, metavar="L",
                        help="Training loss log interval (default: 10)")
    parser.add_argument("--log-file", type=str, default="lii_global_train.log", metavar="F",
                        help="Logging file")
    args = parser.parse_args()

    mode = args.mode
    rnn_type = args.rnn
    input_size = 2
    hidden_size = args.hidden
    layers = args.layers
    epochs = args.epochs
    batch_size = args.batch

    # Set up logging
    BASE_DIR = os.getcwd()
    logging_file = os.path.join(BASE_DIR, "../scripts/{}".format(args.log_file))
    logging.basicConfig(filename=logging_file, filemode="w", level=logging.DEBUG)
    logging.info("Running: {} {} with {} hidden units, {} layers, {} batch size, for {} epochs".format(mode,
                                                                                                       rnn_type,
                                                                                                       hidden_size,
                                                                                                       layers,
                                                                                                       batch_size,
                                                                                                       epochs))

    # Set up device and manual seed
    torch.manual_seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Device being used: {}".format(device))

    # Load data
    test_collection = Collection("../test_data/test_collection")
    posting_lists = []
    posting_length_to_use = 128
    for _, pl in enumerate(test_collection):
        if len(pl[0]) >= posting_length_to_use:
            posting_lists.append(np.array(pl[0], dtype=np.int32))
    posting_lists.sort(key=lambda x: np.shape(x)[0], reverse=True)
    lengths = [len(pl) for pl in posting_lists]
    logging.info("Longest seq: {}".format(max(lengths)))
    logging.info("Shortest seq: {}".format(min(lengths)))
    logging.info("Average seq: {:.2f}".format(np.array(lengths).mean()))
    source_data, source_labels = list_of_tensors(posting_lists, mode)

    # Train model
    lii_rnn = RNNModel(mode, rnn_type, input_size, hidden_size, layers)
    lii_rnn.to(device)
    optimizer = optim.Adam(params=lii_rnn.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True, threshold=10e-6)
    train(device=device,
          model=lii_rnn,
          optimizer=optimizer,
          source_data=source_data,
          source_labels=source_labels,
          lengths=lengths,
          scheduler=scheduler,
          mode=mode,
          epochs=epochs,
          batch_size=batch_size,
          log_interval=args.log)

    # Save model
    logging.info("Model's state_dict:")
    for param_tensor in lii_rnn.state_dict():
        logging.info("{} \t {}".format(param_tensor, lii_rnn.state_dict()[param_tensor].size()))
    MODEL_DIR = os.path.join(BASE_DIR, "../models")
    model_path = os.path.join(MODEL_DIR, "{}_{}_h{}_l{}_e{}.pth".format(mode, rnn_type, hidden_size, layers, epochs))
    logging.info("Saving model to: {}".format(model_path))
    torch.save(lii_rnn.state_dict(), model_path)

    # Generate predictions
    if mode == "doc2doc":
        predictions = evaluate_doc2doc(device, lii_rnn, data, lengths, primer_tokens=1)
    else:
        predictions = evaluate_index2doc(device, lii_rnn, data, lengths)

    # Save delta between prediction and target
    deltas_list = []
    global_max = 0
    for idx, tens in enumerate(predictions):
        delta = (tens.round() - data[idx]).detach().tolist()
        delta_zigzag = [data[idx].size(0)] + list(accumulate([zigzag_encode(int(i)) for i in delta]))
        global_max = max(delta_zigzag[-1], global_max)
        deltas_list.append(np.array(delta_zigzag, dtype=np.uint32))
    global_max_list = [1, global_max]
    deltas_array = np.concatenate([np.array(global_max_list, dtype=np.uint32)] + deltas_list).ravel()
    DELTA_DIR = os.path.join(BASE_DIR, "../deltas")
    delta_file = os.path.join(DELTA_DIR, "{}_{}_h{}_l{}_e{}.docs".format(mode, rnn_type, hidden_size, layers, epochs))
    logging.info("Saving deltas to: {}".format(delta_file))
    with open(delta_file, "wb") as binfile:
        deltas_array.tofile(binfile)


if __name__ == "__main__":
    main()

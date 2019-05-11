import matplotlib.pyplot as plt
import numpy as np

from data_load import load_test_data

TEST_FILE = "test_data/test_collection"


def plot_distributions(pl):
    fig, ax = plt.subplots(figsize=(8, 6))
    pl_length = pl.size()[0]
    indexes = np.arange(pl_length)
    y = pl.cpu().detach().numpy()
    max_y = y.max(axis=0)
    print(max_y)
    ax.plot(indexes, y)
    ax.set_xlabel("Posting list Index")
    ax.set_xlim(0, pl_length)
    ax.set_ylabel("Doc ID")
    ax.set_ylim(0, max_y)
    ax.set_title("Doc IDs Distribution")
    plt.show()


def main():
    posting_lists, pl_lengths = load_test_data(TEST_FILE, 128, shuffling=True)
    pl = posting_lists[0]
    plot_distributions(pl)


if __name__ == '__main__':
    main()

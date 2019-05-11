import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans

from data_load import load_test_data
from collections import defaultdict, OrderedDict

TEST_FILE = "test_data/test_collection"

def plot_distributions(pl, end_points):
    fig, ax = plt.subplots(figsize=(8, 6))
    pl_length = pl.size()[0]
    indexes = np.arange(pl_length)
    y = pl.cpu().detach().numpy()
    max_y = y.max(axis=0)
    ax.plot(indexes, y)
    ax.set_xlabel("Posting list Index")
    ax.set_xlim(0, pl_length)
    ax.set_ylabel("Doc ID")
    ax.set_ylim(0, max_y)
    ax.set_title("Doc IDs Distribution")
    for i in range(len(end_points)):
        ax.plot(end_points[i][0], end_points[i][1])
    plt.show()

def label_points(points, kmeans):
    end_points = OrderedDict()
    i = 0
    prev_label = None
    for label in kmeans.labels_:
        if label not in end_points:
            pt_l = [points[i]]
            end_points[label] = pt_l
        if label != prev_label:
            if prev_label in end_points:
                pt_l = end_points[prev_label]
                pt_l.append(points[i - 1])
        prev_label = label
        i += 1
    end_points[kmeans.labels_[-1]].append(points[-1])
    print(end_points)
    end_point_flatten = []
    for k, ar_l in end_points.items():
        X = []
        X.append(ar_l[0][0])
        X.append(ar_l[1][0])
        y = []
        y.append(ar_l[0][1])
        y.append( ar_l[1][1])
        xy = []
        xy.append(np.array(X))
        xy.append(np.array(y))
        end_point_flatten.append(xy)
    return end_point_flatten

def segments(posting_list, end_points):
    sub_pl = []
    pl_l = posting_list.squeeze().tolist()
    for i in range(len(end_points)):
        idxs = end_points[i][0][0]
        idxe = end_points[i][0][1]
        sub_pl.append((pl_l[idxs:idxe+1], idxe - idxs + 1))
    return sub_pl


def main():
    posting_lists, pl_lengths = load_test_data(TEST_FILE, 1024, shuffling=False)
    pl = posting_lists[0]
    pl_np = pl.numpy().astype(np.int32)
    print(pl_np)
    pl_length = pl.size()[0]
    print(pl_length)
    indexes = np.arange(pl_length)
    points = np.array((indexes, pl_np)).T
    n_clusters = int(pl_length / (5*32))
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    end_points = label_points(points, kmeans)
    print(end_points)
    sub_pl = segments(pl, end_points)
    print(sub_pl)
    plot_distributions(pl, end_points)


if __name__ == '__main__':
    main()

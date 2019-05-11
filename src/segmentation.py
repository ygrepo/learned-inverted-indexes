from collections import OrderedDict

import torch
import numpy as np
from sklearn.cluster import KMeans


class ClusterFinder():
    def __init__(self, posting_list, batch_size):
        assert posting_list is not None
        self.posting_list = posting_list
        self.pl_length = posting_list.size()[1]
        self.indexes = np.arange(self.pl_length)
        self.n_clusters = int(self.pl_length / batch_size)
        self.sub_pl = []

    def label_points(self, points, kmeans):
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
        # print(end_points)
        end_point_flatten = []
        for k, ar_l in end_points.items():
            X = []
            X.append(ar_l[0][0])
            X.append(ar_l[1][0])
            y = []
            y.append(ar_l[0][1])
            y.append(ar_l[1][1])
            xy = []
            xy.append(np.array(X))
            xy.append(np.array(y))
            end_point_flatten.append(xy)
        return end_point_flatten

    def segment(self):
        pl_np = self.posting_list.squeeze().numpy().astype(np.int64)
        points = np.array((self.indexes, pl_np)).T
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(points)
        end_points = self.label_points(points, kmeans)
        sub_pl = []
        for i in range(len(end_points)):
            idxs = end_points[i][0][0]
            idxe = end_points[i][0][1]
            sub_pl.append([self.posting_list[:, idxs:idxe + 1], torch.arange(idxs, idxe + 1)])

        self.sub_pl = sub_pl

    def get_number_batches(self):
        return self.n_clusters

    def __getitem__(self, index):
        return self.sub_pl[index][0], self.sub_pl[index][1]



import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from time import time

from utils import read_from_WADS, analyse_and_visualize

sqrt = math.sqrt


def CRFOR_desnow(pc_data,
                 knn_num=8,
                 low_threshold=-0.4, high_threshold=0.6,
                 intensity_const=1, nbs_mean_d_const=5, pca_const=1,
                 intensity_threshold_constant=2,  # follows LIOR
                 snow_detection_range=30,  # follows LIOR
                 ground_removal=-1.8,  # follows LiSnowNet
                 ):
    snows_idx_dict = dict()

    pts = pc_data[:, :3]
    length = len(pc_data)

    point_ranges = np.linalg.norm(pts, axis=1)

    excluded = pc_data[:, 3] > intensity_threshold_constant  # follows LIOR
    excluded |= point_ranges > snow_detection_range  # follows LIOR

    if ground_removal is not None:
        excluded |= pc_data[:, 2] < ground_removal  # ground removal

    pca = PCA(n_components=3)

    kd_tree = KDTree(pts)
    Dists, Indexes = kd_tree.query(pts.reshape(-1, 3), k=knn_num)

    Features = np.full([length], -1, dtype=np.float32)
    feature_max, feature_min = 0., 1024.

    for i in range(length):
        if excluded[i]:
            continue

        dists, indexes = Dists[i], Indexes[i]

        neighbors_mean_dist = np.mean(dists)

        intensity = pc_data[i][3]

        pca.fit(pts[indexes])
        pca_3rd_dim = pca.explained_variance_ratio_[2]

        # compute and store the "joint feature" for each point
        feature = (pca_const + pca_3rd_dim) * (nbs_mean_d_const + neighbors_mean_dist / (0.1 + point_ranges[i])) / (
                intensity_const + intensity)

        Features[i] = feature
        feature_min = min(feature, feature_min)
        feature_max = max(feature, feature_max)

    # normalize the "joint feature" to [-1, 1], frame-wise
    Features = 2. * (Features - feature_min) / (feature_max - feature_min) - 1

    for i in range(length):
        if not excluded[i]:
            if Features[i] > high_threshold:
                snows_idx_dict[i] = 'f'
            elif Features[i] > low_threshold:
                s = Features[i]

                dists, indexes = Dists[i], Indexes[i]

                for j in range(1, knn_num):
                    f = Features[indexes[j]]
                    if f > high_threshold:
                        f = 2
                    elif f < low_threshold:
                        f = -2
                    s += f / (0.01 + dists[j])

                if s > 0:
                    Features[i] = 2
                    snows_idx_dict[i] = 'f'
                else:
                    Features[i] = -2

    return snows_idx_dict


if __name__ == '__main__':
    # parameters in Tab.1 in our paper
    knn_num, low_threshold, high_threshold, intensity_const, nbs_mean_d_const, pca_const = 8, -0.4, 0.6, 1, 5, 1
    intensity_threshold_constant = 2    # follows LIOR
    snow_detection_range = 30           # follows LIOR
    ground_removal = -1.8               # follows LiSnowNet
    duplicated_removal = True           # follows LiSnowNet

    # todo : change me
    data_path = "sequences/36/velodyne/042244.bin"
    label_path = "sequences/36/labels/042244.label"

    data, label = read_from_WADS(data_path, label_path, duplicated_removal)

    start_time = time()
    snows_idx_dict = CRFOR_desnow(data)
    end_time = time()
    print('run time', end_time - start_time, 's')

    analyse_and_visualize(data, label, snows_idx_dict, visualize_it=True)

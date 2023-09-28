import os

import numpy as np
import math

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from time import time

from utils import read_from_WADS, analyse_and_visualize

sqrt = math.sqrt


# here, CRFOR returns "snows_idx_dict, kd_tree", instead of just "snows_idx_dict"
def CRFOR_basic_desnow(pc_data,
                 knn_num=8,
                 low_threshold=-0.4, high_threshold=0.6,
                 intensity_const=1, nbs_mean_d_const=5, pca_const=1,
                 intensity_threshold_constant=2,    # follows LIOR
                 snow_detection_range=30,           # follows LIOR
                 ground_removal=-1.8,               # follows LiSnowNet
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

    return snows_idx_dict, kd_tree


def CRFOR_temporal_desnow(pc_data,
                 knn_num=8,
                 low_threshold=-0.4, high_threshold=0.6,
                 intensity_const=1, nbs_mean_d_const=5, pca_const=1,
                 intensity_threshold_constant=2,    # follows LIOR
                 snow_detection_range=30,           # follows LIOR
                 ground_removal=-1.8,               # follows LiSnowNet
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

                for pre_frame in range(m):
                    pre_dists, pre_indexes = pre_info[pre_frame]['kdtree'].query(pts[i].reshape(1, -1), k=knn_num)
                    for pre_idx in range(1, knn_num):
                        if pre_indexes[0][pre_idx] in pre_info[pre_frame]['snows'].keys():
                            s += (0.5 ** (m-pre_frame-1)) / (0.01 + pre_dists[0][pre_idx])
                        else:
                            s -= (0.5 ** (m-pre_frame-1)) / (0.01 + pre_dists[0][pre_idx])

                if s > 0:
                    Features[i] = 2
                    snows_idx_dict[i] = 'f'
                else:
                    Features[i] = -2

    return snows_idx_dict, kd_tree


if __name__ == '__main__':
    # parameters in Tab.1 in our paper
    knn_num, low_threshold, high_threshold, intensity_const, nbs_mean_d_const, pca_const = 8, -0.4, 0.6, 1, 5, 1
    intensity_threshold_constant = 2    # follows LIOR
    snow_detection_range = 30           # follows LIOR
    ground_removal = -1.8               # follows LiSnowNet
    duplicated_removal = True           # follows LiSnowNet

    m = 2  # looking at previous 2 frames, except for the first 2 frames

    pre_info = [{'data': None, 'snows': None, 'kdtree': None}] * m

    # todo : change me
    dir_path = "sequences/36"
    # todo : change me
    start_idx, end_idx = 0, 20

    data_dir_path = "{}/velodyne".format(dir_path)
    label_dir_path = "{}/labels".format(dir_path)

    data_paths = os.listdir(data_dir_path)
    label_paths = os.listdir(label_dir_path)

    # for the first m frames, apply basic version of CRFOR
    for idx in range(start_idx, start_idx + m):
        if idx < 0 or idx > len(data_paths):
            raise Exception('idx out of range')

        data, label = read_from_WADS("{}/{}".format(data_dir_path, data_paths[idx]),
                                     "{}/{}".format(label_dir_path, label_paths[idx]),
                                     duplicated_removal)

        print(data_paths[idx], label_paths[idx])

        start_time = time()
        snows_idx_dict, kdtree = CRFOR_basic_desnow(data)
        end_time = time()
        print('run time', end_time - start_time, 's')

        analyse_and_visualize(data, label, snows_idx_dict, visualize_it=False)

        pre_info[idx - start_idx]['data'] = data
        pre_info[idx - start_idx]['snows'] = snows_idx_dict
        pre_info[idx - start_idx]['kdtree'] = kdtree


    # for the following other frames, apply the temporal version, CRFOR-t
    for idx in range(start_idx + m, end_idx):
        if idx < 0 or idx > len(data_paths):
            raise Exception('idx out of range')

        data, label = read_from_WADS("{}/{}".format(data_dir_path, data_paths[idx]),
                                     "{}/{}".format(label_dir_path, label_paths[idx]),
                                     duplicated_removal)

        print(data_paths[idx], label_paths[idx])

        start_time = time()
        snows_idx_dict, kdtree = CRFOR_temporal_desnow(data)
        end_time = time()
        print('run time', end_time - start_time, 's')

        analyse_and_visualize(data, label, snows_idx_dict, visualize_it=False)

        for dummy_idx in range(0, m - 1):
            pre_info[dummy_idx] = pre_info[dummy_idx + 1]
        pre_info[m - 1]['data'] = data
        pre_info[m - 1]['snows'] = snows_idx_dict
        pre_info[m - 1]['kdtree'] = kdtree

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
import scipy
import scipy.io as sio
import os
from sklearn.metrics import average_precision_score
from collections import defaultdict
import pdb
import argparse

def loadfeature_mat(path):
    mat = sio.loadmat(path)
    mat = mat[os.path.split(path)[1].split('.')[0]]
    return mat

def load_test_data(LG_path, FG_path, LP_path, FP_path, CAMG_path, CAMP_path):
    '''
    convert person_id from 0 to 749
    '''
    label_gallery = loadfeature_mat(LG_path)[0]
    GF = loadfeature_mat(FG_path)
    label_probe = loadfeature_mat(LP_path)[0]
    PF = loadfeature_mat(FP_path)
    GCAM = loadfeature_mat(CAMG_path)[0]
    PCAM = loadfeature_mat(CAMP_path)[0]
    pdb.set_trace()

    D = pairwise_distances(GF, PF, metric='euclidean', n_jobs=1)
    D = np.transpose(D)

    unique_labels = np.unique(np.r_[label_gallery,label_probe])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    PL = np.asarray([labels_map[l] for l in label_probe])
    GL = np.asarray([labels_map[l] for l in label_gallery])
    return D, PL, GL, PCAM, GCAM

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, topk=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    pdb.set_trace()

    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        #pdb.set_trace()
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    print (num_valid_queries)
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_gallery_path", help='xxx')
    parser.add_argument("--feature_gallery_path", help='xxx')
    parser.add_argument("--label_probe_path", help='xxx')
    parser.add_argument("--feature_probe_path", help='xxx')
    parser.add_argument("--cam_gallery_path", help='xxx')
    parser.add_argument("--cam_probe_path", help='xxx')
    args = parser.parse_args()
    # print(args.feature_gallery_path) 
    # label_gallery_path = '/home/zhangkaicheng/data/test_gallery_labels.mat'
    # feature_gallery_path = '/home/zhangkaicheng/data/test_gallery_features.mat'
    # label_probe_path = '/home/zhangkaicheng/data/test_probe_labels.mat'
    # feature_probe_path = '/home/zhangkaicheng/data/test_probe_features.mat'
    # cam_gallery_path = '/home/zhangkaicheng/data/testCAM.mat'
    # cam_probe_path = '/home/zhangkaicheng/data/queryCAM.mat'
    DIST, PL, GL, PCAM, GCAM = load_test_data(args.label_gallery_path, 
    	args.feature_gallery_path, 
    	args.label_probe_path, 
    	args.feature_probe_path,
    	args.cam_gallery_path,
    	args.cam_probe_path
    	)
    ans = cmc(DIST, PL, GL, PCAM, GCAM, first_match_break=True)
    # ans = cmc(DIST, separate_camera_set=True)
    print(ans)

if __name__ == "__main__":
    main()

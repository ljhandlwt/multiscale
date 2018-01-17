import tensorflow as tf
import pdb
import os
import time
import re
import numpy as np
from scipy import io

probe_features_path_299 = './59286/test_probe_features.mat'
gallery_features_path_299 = './59286/test_gallery_features.mat'
probe_features_path_225 = './58626/test_probe_features.mat'
gallery_features_path_225 = './58626/test_gallery_features.mat'

probe_labels_path_299 = './59286/test_probe_labels.mat'
gallery_labels_path_299 = './59286/test_gallery_labels.mat'
probe_labels_path_225 = './58626/test_probe_labels.mat'
gallery_labels_path_225 = './58626/test_gallery_labels.mat'

probe1_features = io.loadmat(probe_features_path_299)
probe2_features = io.loadmat(probe_features_path_225)
probe1_labels = io.loadmat(probe_labels_path_299)
probe2_labels = io.loadmat(probe_labels_path_225)

probe1_features = probe1_features['test_probe_features']
probe2_features = probe2_features['test_probe_features']
probe1_labels = probe1_labels['test_probe_labels']
probe2_labels = probe2_labels['test_probe_labels']

gallery1_features = io.loadmat(gallery_features_path_299)
gallery2_features = io.loadmat(gallery_features_path_225)
gallery1_labels = io.loadmat(gallery_labels_path_299)
gallery2_labels = io.loadmat(gallery_labels_path_225)

gallery1_features = gallery1_features['test_gallery_features']
gallery2_features = gallery2_features['test_gallery_features']
gallery1_labels = gallery1_labels['test_gallery_labels']
gallery2_labels = gallery2_labels['test_gallery_labels']

probe_features_new = np.zeros((probe1_labels.shape[1], 4096), dtype='float32')

for i in range(probe1_labels.shape[1]):
	for j in range(probe2_labels.shape[1]):
		if (probe1_labels[0][i] == probe2_labels[0][j]):
			new_features = np.hstack([probe1_features[i], probe2_features[j]])
			probe_features_new[i] = new_features

gallery_features_new = np.zeros((gallery1_labels.shape[1], 4096), dtype='float32')

for i in range(gallery1_labels.shape[1]):
	for j in range(gallery2_labels.shape[1]):
		if (gallery1_labels[0][i] == gallery2_labels[0][j]):
			new_features = np.hstack([gallery1_features[i], gallery2_features[j]])
			gallery_features_new[i] = new_features

io.savemat('./concat_probe_features.mat', {'concat_probe_features': probe_features_new})
io.savemat('./concat_gallery_features.mat', {'concat_gallery_features': gallery_features_new})
io.savemat('./concat_probe_labels.mat', {'concat_probe_labels': probe1_labels})
io.savemat('./concat_gallery_labels.mat', {'concat_gallery_labels': gallery1_labels})

pdb.set_trace()

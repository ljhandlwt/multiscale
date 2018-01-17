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

pdb.set_trace()


import tensorflow as tf
import pdb
import os
import time
import re
import numpy as np
from scipy import io

_probe_features_path = './59286/test_probe_features.mat'
299_gallery_features_path = './59286/test_gallery_features.mat'
225_probe_features_path = './58626/test_probe_features.mat'
225_gallery_features_path = './58626/test_gallery_features.mat'

299_probe_labels_path = './59286/test_probe_labels.mat'
299_gallery_labels_path = './59286/test_gallery_labels.mat'
225_probe_labels_path = './58626/test_probe_labels.mat'
225_gallery_labels_path = './58626/test_gallery_labels.mat'

probe1_features = io.loadmat(299_probe_features_path)
probe2_features = io.loadmat(225_probe_features_path)
probe1_labels = io.loadmat(299_probe_labels_path)
probe2_labels = io.loadmat(225_probe_labels_path)

probe1_features = probe1_features['test_probe_features']
probe2_features = probe2_features['test_probe_features']
probe1_labels = probe1_labels['test_probe_labels']
probe2_labels = probe2_labels['test_probe_labels']

pdb.set_trace()


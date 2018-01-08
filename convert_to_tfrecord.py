from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#encoding=utf-8

import skimage
import skimage.io
import sys
import os
import multiprocessing
import numpy as np 
import tensorflow as tf
import random
import pdb

#setting path
input_path = '/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501/bounding_box_train'
output_path = '/world/data-gpu-94/sysu-reid/zhangkaicheng/Market-1501-tfrecord/bounding_box_train'

num_workers = 4

num_file = 4

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def worker(args, dict_labels):
    img_names, output_file = args
    
    writer = tf.python_io.TFRecordWriter(output_file)

    imgs = [skimage.io.imread(os.path.join(input_path,name)) for name in img_names]
    labels = [int(name[:name.find('_')]) for name in file_names]

    for i in range(len(labels)):
        img = imgs[i]
        label = dict_labels[labels[i]]
        row, col, _ = img.shape

        example = tf.train.Example(features=tf.train.Features(feature={
            'img_height': _int64_feature(int(row)),
            'img_width': _int64_feature(int(col)),
            'img': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
            }))
        writer.write(example.SerializeToString())

    writer.close()

    return 0

def main():
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_names = os.listdir(input_path)
    file_names = [name for name in file_names if name.endswith('jpg')]
    random.shuffle(file_names)

    labels = [int(name[:name.find('_')]) for name in file_names]
    unique_labels = list(set(labels))
    unique_labels.sort()
    dict_labels = dict(zip(unique_labels, range(len(unique_labels))))

    n = len(file_names)//num_file

    pool_args = []
    for i in range(num_file):
        p_args = []
        ibeg = i * n
        iend = ibeg + n
        p_args.append(file_names[ibeg:iend])
        p_args.append(os.path.join(output_path, 'm_{}.tfrecords'.format(i)))
        pool_args.append(p_args)

    pool = multiprocessing.Pool(num_workers)

    #pdb.set_trace()

    pool.map(worker, pool_args, dict_labels)
    pool.close()
    pool.join()
    #map(worker, pool_args)

if __name__ == "__main__":
    main()

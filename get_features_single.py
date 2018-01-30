from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb
import os
import time
import re
import numpy as np
import sys
from scipy import io

from deployment import model_deploy

from nets import my_model_single as my_model

# jh-future:it needs to be add to tf.app.flags
os.environ["CUDA_VISIBLE_DEVICES"]="2"

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1e-8, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'probe_dataset_dir', None, 'The directory where the probe dataset files are stored.')

tf.app.flags.DEFINE_string(
    'gallery_dataset_dir', None, 'The directory where the gallery dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

# jh-future:you will need a last_step to restore from any step you like, not just the last step

tf.app.flags.DEFINE_integer('origin_height', 128, 'origin height of image')

tf.app.flags.DEFINE_integer('origin_width', 64, 'origin width of image')

tf.app.flags.DEFINE_integer('origin_channel', 3, 'origin channel of image')

tf.app.flags.DEFINE_integer('num_classes', 751, 'num of classes')

tf.app.flags.DEFINE_integer('scale_size', 299, 'size of scale in single model')

tf.app.flags.DEFINE_integer(
    'ckpt_num', None, 'The number of ckpt model.')

#####################
# Dir Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_dir', None,
    'dir to save checkpoint')

tf.app.flags.DEFINE_string(
    'pretrain_path', None,
    'path to load pretrain model')

tf.app.flags.DEFINE_string(
  'log_dir', None, 'dir of summar')

FLAGS = tf.app.flags.FLAGS

class Get_feature(object):
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        with tf.Graph().as_default():
            self.init_probe_batch()
            self.init_gallery_batch()
            self.init_network()
            self.get_feature()

    def init_probe_batch(self):
        deploy_config = model_deploy.DeploymentConfig()

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
            self.global_step = global_step

        probe_tfrecord_list = os.listdir(FLAGS.probe_dataset_dir)
        probe_tfrecord_list = [os.path.join(FLAGS.probe_dataset_dir, name) for name in probe_tfrecord_list if name.endswith('tfrecords')]
        probe_file_queue = tf.train.string_input_producer(probe_tfrecord_list, num_epochs=1)

        reader = tf.TFRecordReader()
        _, probe_serialized_example = reader.read(probe_file_queue)

        probe_features = tf.parse_single_example(probe_serialized_example,features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img' : tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'cam': tf.FixedLenFeature([], tf.int64)
            })

        probe_img = tf.decode_raw(probe_features['img'], tf.uint8)
        probe_img_height = tf.cast(probe_features['img_height'], tf.int32)
        probe_img_width = tf.cast(probe_features['img_width'], tf.int32)
        probe_img = tf.reshape(probe_img, tf.stack([FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel]))
        probe_img = tf.image.convert_image_dtype(probe_img, dtype=tf.float32)

        probe_label = probe_features['label']
        probe_cam = probe_features['cam']
        probe_images, probe_labels, probe_cams = tf.train.batch([probe_img, probe_label, probe_cam],
            batch_size = FLAGS.batch_size,
            capacity = 3000,
            num_threads = 4,
            allow_smaller_final_batch=True
        )

        self.deploy_config = deploy_config
        self.global_step = global_step

        self.probe_images = probe_images
        self.probe_labels = probe_labels
        self.probe_cams = probe_cams

    def init_gallery_batch(self):
        deploy_config = model_deploy.DeploymentConfig()

        gallery_tfrecord_list = os.listdir(FLAGS.gallery_dataset_dir)
        gallery_tfrecord_list = [os.path.join(FLAGS.gallery_dataset_dir, name) for name in gallery_tfrecord_list if name.endswith('tfrecords')]
        gallery_file_queue = tf.train.string_input_producer(gallery_tfrecord_list, num_epochs=1)

        reader = tf.TFRecordReader()
        _, gallery_serialized_example = reader.read(gallery_file_queue)

        gallery_features = tf.parse_single_example(gallery_serialized_example,features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img' : tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'cam': tf.FixedLenFeature([], tf.int64)
            })

        gallery_img = tf.decode_raw(gallery_features['img'], tf.uint8)
        gallery_img_height = tf.cast(gallery_features['img_height'], tf.int32)
        gallery_img_width = tf.cast(gallery_features['img_width'], tf.int32)
        gallery_img = tf.reshape(gallery_img, tf.stack([FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel]))
        gallery_img = tf.image.convert_image_dtype(gallery_img, dtype=tf.float32)

        gallery_label = gallery_features['label']
        gallery_cam = gallery_features['cam']
        gallery_images, gallery_labels, gallery_cams = tf.train.batch([gallery_img, gallery_label, gallery_cam],
            batch_size = FLAGS.batch_size,
            capacity = 3000,
            num_threads = 4,
            allow_smaller_final_batch=True
        )

        self.gallery_images = gallery_images
        self.gallery_labels = gallery_labels
        self.gallery_cams = gallery_cams

    def init_network(self):
        # jh-future:sizes can be add into tf.app.flags
        network = my_model.MyInception(
            FLAGS.num_classes-FLAGS.labels_offset,
            [FLAGS.scale_size],
            FLAGS.model_name,
            is_training=False
        )

        self.network = network

    def get_feature(self):
        # sess
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement=True #allow cpu calc when gpu can't
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        self.sess = sess

        # summary
        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
        summary_op = tf.summary.merge(summaries, name='summary_op')
        self.summary_op = summary_op

        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)
        self.summary_writer = summary_writer

        # saver
        saver = tf.train.Saver()
        self.saver = saver

        # load model
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        last_step = self.load_model()

        # multi-thread-read
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        self.coord = coord
        self.threads = threads

        # init vars
        probe_img_features = []
        probe_img_label = []
        probe_img_cam = []

        for i in range(99999):
            try:
                batch = self.sess.run([self.probe_images, self.probe_labels, self.probe_cams])
            except Exception as e:
               break

            feed = {
                self.network.image:batch[0]
            }
            probe_calc_obj = [self.network.feature]

            probe_calc_ans = self.sess.run(probe_calc_obj, feed_dict=feed)

            probe_img_features.append(np.squeeze(probe_calc_ans[0], axis=None))
            probe_img_label.append(batch[1])
            probe_img_cam.append(batch[2])

        probe_img_features = np.concatenate(probe_img_features, axis=0)
        probe_img_label = np.concatenate(probe_img_label, axis=0)
        probe_img_cam = np.concatenate(probe_img_cam, axis=0)

        print (probe_img_features.shape)

        file_path = str(FLAGS.ckpt_num)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        io.savemat(file_path + '/test_probe_features.mat', {'test_probe_features': probe_img_features})
        io.savemat(file_path + '/test_probe_labels.mat', {'test_probe_labels': probe_img_label})
        io.savemat(file_path + '/queryCAM.mat', {'queryCAM': probe_img_cam})

        # init vars
        gallery_img_features = []
        gallery_img_label = []
        gallery_img_cam = []

        for i in range(99999):
            try:
                batch = self.sess.run([self.gallery_images, self.gallery_labels, self.gallery_cams])
            except Exception as e:
               break

            feed = {
                self.network.image:batch[0]
            }
            gallery_calc_obj = [self.network.feature]

            gallery_calc_ans = self.sess.run(gallery_calc_obj, feed_dict=feed)

            gallery_img_features.append(np.squeeze(gallery_calc_ans[0], axis=None))
            gallery_img_label.append(batch[1])
            gallery_img_cam.append(batch[2])

        gallery_img_features = np.concatenate(gallery_img_features, axis=0)
        gallery_img_label = np.concatenate(gallery_img_label, axis=0)
        gallery_img_cam = np.concatenate(gallery_img_cam, axis=0)

        print (gallery_img_features.shape)

        io.savemat(file_path + '/test_gallery_features.mat', {'test_gallery_features': gallery_img_features})
        io.savemat(file_path + '/test_gallery_labels.mat', {'test_gallery_labels': gallery_img_label})
        io.savemat(file_path + '/testCAM.mat', {'testCAM': gallery_img_cam})

        # end
        self.summary_writer.close()
        self.coord.request_stop()
        self.coord.join(self.threads)

    def load_model(self):
        # return num of last-batch
        # if no checkpoint, return -1
        # pdb.set_trace()
        if os.path.exists(FLAGS.checkpoint_dir):
            filenames = os.listdir(FLAGS.checkpoint_dir)
            filenames = [name for name in filenames if name.endswith('index')]
            if len(filenames) > 0:
                # pattern = r'model\.ckpt\-(\d+)\.index'
                # nums = [int(re.search(pattern, name).groups()[0]) for name in filenames]
                max_num = FLAGS.ckpt_num

                self.saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-{}'.format(max_num)))
                print("[JH]use checkpoint-{} weights".format(max_num))
                return max_num
        if os.path.exists(FLAGS.pretrain_path):
            self.network.load_pretrain_model(self.sess, FLAGS.pretrain_path)
            print("[JH]use pretrain init weights")
            return -1

        print("[JH]use random init weights")
        return -1

def main(_):
    Get_feature()

if __name__ == '__main__':
    tf.app.run()

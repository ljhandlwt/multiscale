from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb
import os
import time
import re
import sys

from deployment import model_deploy

from nets import my_model

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_model_summary_secs', 600,
    'The frequency with which the model is saved and summaries are saved, in seconds.')

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

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

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

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

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
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to train.')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_integer('origin_height', 128, 'origin height of image')

tf.app.flags.DEFINE_integer('origin_width', 64, 'origin width of image')

tf.app.flags.DEFINE_integer('origin_channel', 3, 'origin channel of image')

tf.app.flags.DEFINE_integer('num_classes', 751, 'num of classes')

tf.app.flags.DEFINE_string('GPU_use', '0', 'number of GPU to use')

tf.app.flags.DEFINE_integer('size_branch_0', 299, 'size of branch0')

tf.app.flags.DEFINE_integer('size_branch_1', 225, 'size of branch1')

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

tf.app.flags.DEFINE_string(
  'pretrain_branch_0_path', None, 'pretrain ckpt of 299 model')

tf.app.flags.DEFINE_string(
  'pretrain_branch_1_path', None, 'pretrain ckpt of 225 model')

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.GPU_use

def _configure_learning_rate(num_samples_per_epoch, global_step):
  """Configures the learning rate.

  Args:
    num_samples_per_epoch: The number of samples in each epoch of training.
    global_step: The global_step tensor.

  Returns:
    A `Tensor` representing the learning rate.

  Raises:
    ValueError: if
  """
  decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                    FLAGS.num_epochs_per_decay)
  if FLAGS.sync_replicas:
    decay_steps /= FLAGS.replicas_to_aggregate

  if FLAGS.learning_rate_decay_type == 'exponential':
    return tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'fixed':
    return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
  elif FLAGS.learning_rate_decay_type == 'polynomial':
    return tf.train.polynomial_decay(FLAGS.learning_rate,
                                     global_step,
                                     decay_steps,
                                     FLAGS.end_learning_rate,
                                     power=1.0,
                                     cycle=False,
                                     name='polynomial_decay_learning_rate')
  else:
    raise ValueError('learning_rate_decay_type [%s] was not recognized',
                     FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
  """Configures the optimizer used for training.

  Args:
    learning_rate: A scalar or `Tensor` learning rate.

  Returns:
    An instance of an optimizer.

  Raises:
    ValueError: if FLAGS.optimizer is not recognized.
  """
  if FLAGS.optimizer == 'adadelta':
    optimizer = tf.train.AdadeltaOptimizer(
        learning_rate,
        rho=FLAGS.adadelta_rho,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'adagrad':
    optimizer = tf.train.AdagradOptimizer(
        learning_rate,
        initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
  elif FLAGS.optimizer == 'adam':
    # pdb.set_trace()
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=FLAGS.adam_beta1,
        beta2=FLAGS.adam_beta2,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    optimizer = tf.train.FtrlOptimizer(
        learning_rate,
        learning_rate_power=FLAGS.ftrl_learning_rate_power,
        initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
        l1_regularization_strength=FLAGS.ftrl_l1,
        l2_regularization_strength=FLAGS.ftrl_l2)
  elif FLAGS.optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(
        learning_rate,
        momentum=FLAGS.momentum,
        name='Momentum')
  elif FLAGS.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=FLAGS.rmsprop_decay,
        momentum=FLAGS.rmsprop_momentum,
        epsilon=FLAGS.opt_epsilon)
  elif FLAGS.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
  return optimizer

class que(object):
    def __init__(self, max_size=1000):
        self.arr = []
        self.max_size = max_size
   
    def append(self, num):
        self.arr.append(num)
        if len(self.arr) > self.max_size :
            self.arr.pop(0)

    def avg(self):
        if not len(self.arr):
            return 0
        return sum(self.arr)/len(self.arr)

class Trainer(object):
    def __init__(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        with tf.Graph().as_default():
            self.init_batch()
            self.init_network()
            self.init_opt()

            self.train()

    def init_batch(self):
        deploy_config = model_deploy.DeploymentConfig()

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()
            self.global_step = global_step

        tfrecord_list = os.listdir(FLAGS.dataset_dir)
        tfrecord_list = [os.path.join(FLAGS.dataset_dir, name) for name in tfrecord_list if name.endswith('tfrecords')]
        file_queue = tf.train.string_input_producer(tfrecord_list)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(file_queue)

        features = tf.parse_single_example(serialized_example,features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img' : tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64)
            })

        img = tf.decode_raw(features['img'], tf.uint8)
        img_height = tf.cast(features['img_height'], tf.int32)
        img_width = tf.cast(features['img_width'], tf.int32)
        img = tf.reshape(img, tf.stack([FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel]))
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.random_flip_left_right(img)

        label = features['label']
        images, labels = tf.train.shuffle_batch([img, label],
            batch_size = FLAGS.batch_size,
            capacity = 3000,
            min_after_dequeue = 1000,
            num_threads = 4
        )
        labels = tf.one_hot(labels, FLAGS.num_classes-FLAGS.labels_offset)

        #self.dataset = dataset
        self.deploy_config = deploy_config
        self.global_step = global_step

        self.images = images
        self.labels = labels

    def init_network(self):
        # jh-future:sizes can be add into tf.app.flags
        network = my_model.MyInception(
            FLAGS.num_classes-FLAGS.labels_offset,
            [FLAGS.size_branch_0, FLAGS.size_branch_1],
            FLAGS.model_name,
            is_training=True
        )

        self.network = network

    def init_opt(self):
        with tf.device(self.deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(12000, self.global_step) #5000 is a discarded para
            optimizer = _configure_optimizer(learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)

        self.learning_rate = learning_rate
        self.optimizer = optimizer

        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grad = tf.gradients(self.network.loss, variables)
        bn_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = [self.optimizer.apply_gradients(zip(grad,variables))] + bn_op

        bn_op_branch_0 = [bn for bn in bn_op if bn.name.startswith('inception_v3/branch_0')]
        variables_branch_0 = [var for var in variables if var.name.startswith('inception_v3/branch_0')]
        grad_branch_0 = tf.gradients(self.network.loss_branch_0, variables_branch_0)
        self.train_op_branch_0 = [self.optimizer.apply_gradients(zip(grad_branch_0,variables_branch_0))] + bn_op_branch_0

        bn_op_branch_1 = [bn for bn in bn_op if bn.name.startswith('inception_v3/branch_1')]
        variables_branch_1 = [var for var in variables if var.name.startswith('inception_v3/branch_1')]
        grad_branch_1 = tf.gradients(self.network.loss_branch_1, variables_branch_1)
        self.train_op_branch_1 = [self.optimizer.apply_gradients(zip(grad_branch_1,variables_branch_1))] + bn_op_branch_1
    
    def train(self):
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
        saver = tf.train.Saver(max_to_keep=10)
        self.saver = saver

        # load model
        sess.run(tf.global_variables_initializer())
        last_step = self.load_model()

        # multi-thread-read
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord,sess=self.sess)
        self.coord = coord
        self.threads = threads

        # init vars
        st_time = time.time()
        last_save_time = st_time

        sum_joint_loss = que()
        sum_joint_acc = que()
        sum_loss_branch_0 = que()
        sum_loss_branch_1 =que()
        sum_acc_branch_0 = que()
        sum_acc_branch_1 = que()

        # train
        for i in range(last_step+1, FLAGS.max_number_of_steps):
            # summary and save model
            timeout = time.time() - last_save_time > FLAGS.save_model_summary_secs
            if timeout:
                last_save_time = time.time()

            # batch
            batch = self.sess.run([self.images, self.labels])

            # feed
            feed = {
                self.network.image:batch[0],
                self.network.label:batch[1]
            }
            # calc_obj
            calc_obj = [self.train_op, self.network.loss, self.network.joint_acc, 
            	self.network.sub_models[0].loss, self.network.sub_models[1].loss, 
            	self.network.sub_models[0].acc, self.network.sub_models[1].acc]
            if timeout:
                calc_obj.append(self.summary_op)
            calc_obj_branch_0 = [self.train_op_branch_0]
            calc_obj_branch_1 = [self.train_op_branch_1]

            # run
            calc_ans = self.sess.run(calc_obj, feed_dict=feed)
            self.sess.run(calc_obj_branch_0, feed_dict=feed)
            self.sess.run(calc_obj_branch_1, feed_dict=feed)

            # loss-acc
            sum_joint_loss.append(calc_ans[1])
            sum_joint_acc.append(calc_ans[2])
            sum_loss_branch_0.append(calc_ans[3])
            sum_loss_branch_1.append(calc_ans[4])
            sum_acc_branch_0.append(calc_ans[5])
            sum_acc_branch_1.append(calc_ans[6])
            # print info
            if i % FLAGS.log_every_n_steps == 0:
                avg_joint_loss = sum_joint_loss.avg()
                avg_joint_acc = sum_joint_acc.avg() / FLAGS.batch_size
                avg_loss_branch_0 = sum_loss_branch_0.avg()
                avg_loss_branch_1 = sum_loss_branch_1.avg()
                avg_acc_branch_0 = sum_acc_branch_0.avg() / FLAGS.batch_size
                avg_acc_branch_1 = sum_acc_branch_1.avg() / FLAGS.batch_size
                print("[%d] joint_loss: %.5f joint_acc: %.5f"%(i, avg_joint_loss, avg_joint_acc))
                print("[%d] loss_branch_0: %.5f acc_branch_0: %.5f"%(i, avg_loss_branch_0, avg_acc_branch_0))
                print("[%d] loss_branch_1: %.5f acc_branch_1: %.5f"%(i, avg_loss_branch_1, avg_acc_branch_1))
            	print("-------------------------------------------------------------------------")

            # save model and summary
            if timeout:
                save_path = os.path.join(FLAGS.checkpoint_dir,'model.ckpt')
                self.saver.save(self.sess, save_path, global_step=i)
                self.summary_writer.add_summary(calc_ans[7], global_step=i)

        # end
        self.summary_writer.close()
        self.coord.request_stop()
        self.coord.join(self.threads)

    def load_model(self):
        # return num of last-batch
        # if no checkpoint, return -1
        if os.path.exists(FLAGS.checkpoint_dir):
            filenames = os.listdir(FLAGS.checkpoint_dir)
            filenames = [name for name in filenames if name.endswith('index')]
            if len(filenames) > 0:
                pattern = r'model\.ckpt\-(\d+)\.index'
                nums = [int(re.search(pattern, name).groups()[0]) for name in filenames]
                max_num = max(nums)

                self.saver.restore(self.sess, os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-{}'.format(max_num)))
                print("[JH]use checkpoint-{} weights".format(max_num))
                return max_num
        else:
            self.network.load_pretrain_model(self.sess, 
                [FLAGS.pretrain_branch_0_path, FLAGS.pretrain_branch_1_path])
            print("[ZKC]use pretrain init weights")
            return -1

        print("[JH]use random init weights")
        return -1

def main(_):
    Trainer()

if __name__ == '__main__':
    tf.app.run()

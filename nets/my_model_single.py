from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pdb

from nets import inception_v3

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim

class BaseModel(object):
    def __init__(self):
        '''
        we should declear:
        self.logits
        self.end_points
        self.loss
        self.acc
        '''

class MyInception(BaseModel):
    def __init__(self, num_classes, sizes, scope, is_training=True):
        self.num_classes = num_classes
        self.sizes = sizes
        self.scope = scope
        self.is_training = is_training

        with tf.variable_scope(scope):
            self.init_input()
            with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
                self.init_network()
            self.init_loss()

    def init_input(self):
        self.image = tf.placeholder(tf.float32, [None, FLAGS.origin_height, FLAGS.origin_width, FLAGS.origin_channel])
        self.label = tf.placeholder(tf.float32, [None, self.num_classes])

    def init_network(self):
        self.sub_models = []
        for i,s in enumerate(self.sizes):
            sub_model = SubIncption([self.image,self.label],
                self.num_classes, s, 'branch_{}'.format(i), is_training=self.is_training)
            self.sub_models.append(sub_model)

        if len(self.sizes) > 1:
            joint_scope = 'joint'

            with tf.variable_scope(joint_scope):
                self.feature = tf.concat([self.sub_models[0].end_points['AvgPool_1a'],
                    self.sub_models[1].end_points['AvgPool_1a']], axis=-1)
                x = tf.concat([model.logits for model in self.sub_models], axis=-1)
                x = slim.fully_connected(x, self.num_classes, normalizer_fn=None, scope='joint_fc')
                self.joint_logits = x
                self.joint_pred = tf.nn.softmax(x)

                corr_pred = tf.equal(tf.argmax(self.label,1), tf.argmax(self.joint_logits,1))
                self.joint_acc = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

                self.logits = self.joint_logits
                self.end_points = {}
                for model in self.sub_models:
                    for end_point in model.end_points:
                        self.end_points[model.scope+'/'+end_point] = model.end_points[end_point]
                self.end_points[joint_scope+'/Logits'] = self.joint_logits
                self.end_points[joint_scope+'/Predictions'] = self.joint_pred


            tf.summary.histogram('activations/%s/%s'%(self.scope,'Logits'), self.joint_logits)
            tf.summary.scalar('sparsity/%s/%s'%(self.scope,'Logits'), tf.nn.zero_fraction(self.joint_logits))
            tf.summary.histogram('activations/%s/%s'%(self.scope,'Predictions'), self.joint_pred)
            tf.summary.scalar('sparsity/%s/%s'%(self.scope,'Predictions'), tf.nn.zero_fraction(self.joint_pred))

            tf.summary.scalar('acc/%s' % self.scope, self.joint_acc)
        else:
            self.logits = self.sub_models[0].logits
            self.end_points = self.sub_models[0].end_points
            self.feature = self.sub_models[0].end_points['AvgPool_1a']

    def init_loss(self):
        cross_entropy = tf.reduce_sum([model.loss for model in self.sub_models])

        if len(self.sizes) > 1:
            joint_cross_entropy = -tf.reduce_sum(self.label*tf.log(self.joint_pred+FLAGS.opt_epsilon), axis=1)
            joint_cross_entropy = tf.reduce_mean(joint_cross_entropy)
            cross_entropy = cross_entropy + joint_cross_entropy

            tf.summary.scalar('losses/%s_joint' % self.scope, joint_cross_entropy)

        regular_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regularizers = tf.add_n(regular_vars)

        self.loss = cross_entropy + FLAGS.weight_decay * regularizers

        tf.summary.scalar('losses/%s_cross_entropy' % self.scope, cross_entropy)
        tf.summary.scalar('losses/%s_regularizers' % self.scope, regularizers)
        tf.summary.scalar('losses/%s' % self.scope, self.loss)

    def load_pretrain_model(self, sess, path):
        # make sure self.scope is the root scope
        for model in self.sub_models:
            model.load_pretrain_model(sess, path, self.scope)

class SubIncption(BaseModel):
    def __init__(self, input, num_classes, size, scope, is_training=True):
        self.image = input[0]
        self.label = input[1]
        self.num_classes = num_classes
        self.size = size
        self.scope = scope
        self.is_training = is_training

        with tf.variable_scope(self.scope):
            self.init_network()
            self.init_loss()

    def init_network(self):
        x = self.image
        x = tf.image.resize_images(x, [self.size,self.size], 0) #0 mean bilinear
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)

        logits,end_points = inception_v3.inception_v3(x,
            num_classes=self.num_classes,
            is_training=self.is_training,
        )

        self.logits = logits
        self.pred = end_points['Predictions']
        self.end_points = end_points

        corr_pred = tf.equal(tf.argmax(self.label,1), tf.argmax(self.logits,1))
        self.acc = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

        for end_point in self.end_points:
            x = self.end_points[end_point]
            tf.summary.histogram('activations/%s/%s'%(self.scope,end_point), x)
            tf.summary.scalar('sparsity/%s/%s'%(self.scope,end_point), tf.nn.zero_fraction(x))

        tf.summary.scalar('acc/%s' % self.acc, self.acc)

    def init_loss(self):
        cross_entropy = -tf.reduce_sum(self.label*tf.log(self.pred+FLAGS.opt_epsilon), axis=1)
        self.loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('losses/%s' % self.scope, self.loss)

    def load_pretrain_model(self, sess, path, father_scope):
        '''
        in pretrain, name like:InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/weights
        in our model, name like:inception_v3/branch_0/InceptionV3/Mixed_7b/Branch_2/Conv2d_0d_3x1/weights:0
        so, model_name = inception_v3/branch_0/ + pretrain_name + ':0'
                       = father_scope/sub_model_scope/pretrain_name:0

        note:
        some vars can't be load like final-fc
        in pretrain, name of these vars like:InceptionV3/AuxLogits/Conv2d_2b_1x1/biases
                                             InceptionV3/Logits/Conv2d_2b_1x1/biases
        they all have prefix like InceptionV3/AuxLogits/ or InceptionV3/Logits/
        '''
        scope = father_scope + '/' + self.scope + '/'
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        d = {}
        for var in variables:
            name = var.name.replace(scope, '').replace(':0', '')
            if name.startswith('InceptionV3/AuxLogits') or name.startswith('InceptionV3/Logits'):
                continue
            d[name] = var

        saver = tf.train.Saver(d)
        saver.restore(sess, path)

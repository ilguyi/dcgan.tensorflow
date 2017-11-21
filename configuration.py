"""Configuration
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


##################
# Training Flags #
##################
tf.app.flags.DEFINE_string('train_dir',
                           '',
                           'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('max_steps',
                            10000,
                            'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('save_steps',
                            5000,
                            'The step per saving model.')

#################
# Dataset Flags #
#################
tf.app.flags.DEFINE_string('dataset_dir',
                           None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('num_examples',
                            100,
                            'The number of samples in total dataset.')

########################
# Learning rate policy #
########################
tf.app.flags.DEFINE_float('initial_learning_rate',
                          0.0001,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay',
                          100,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor',
                          0.5,
                          'Learning rate decay factor.')

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_string('optimizer',
                           'adam',
                           'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
                           '"ftrl", "momentum", "sgd" or "rmsprop"')
tf.app.flags.DEFINE_float('adam_beta1',
                          0.9,
                          'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2',
                          0.999,
                          'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('adam_epsilon',
                          1e-08,
                          'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('l2_decay',
                          0.0004,
                          'L2 regularization factor for the optimizer.')

########################
# Moving average decay #
########################
tf.app.flags.DEFINE_float('MOVING_AVERAGE_DECAY',
                          0.9999,
                          'Moving average decay.')

####################
# Checkpoint Flags #
####################
tf.app.flags.DEFINE_string('checkpoint_dir',
                           '',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model.')

####################
# Generating Flags #
####################
tf.app.flags.DEFINE_boolean('make_gif',
                            False,
                            'Whether make gif or not.')
tf.app.flags.DEFINE_integer('seed',
                            0,
                            'The seed number.')


FLAGS = tf.app.flags.FLAGS


def hyperparameters_dir(input_dir):
  hp_dir = os.path.join(input_dir, FLAGS.optimizer)
  if FLAGS.optimizer == "adam":
    hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.adam_beta1)
  hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.initial_learning_rate)
  hp_dir = os.path.join(hp_dir, '%.1E' % FLAGS.l2_decay)
  print(hp_dir)

  return hp_dir


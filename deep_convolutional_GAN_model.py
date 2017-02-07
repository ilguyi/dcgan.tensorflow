from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import copy

import tensorflow as tf

import ops
import image_processing

slim = tf.contrib.slim
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

FLAGS = tf.app.flags.FLAGS



class DeepConvGANModel(object):
  """Deep Convolutional Generative Adversarial Networks
  implementation based on http://arxiv.org/abs/1511.06434

  "Unsupervised Representation Learning with
  Deep Convolutional Generative Adversarial Networks"
  Alec Radford, Luke Metz and Soumith Chintala
  """

  def __init__(self):
    """Basic setup.
    """
    #assert mode in ["train", "generate"]
    #self.mode = mode

    # A int32 scalar value;
    self.random_z_size = 100

    print('complete initializing model.')



  def Generator(self, random_z):
    """Generator setup.

    Args:
      random_z: np.ndarray random vector (latent code)
    """
    with tf.variable_scope('Generator'):
      batch_norm_params = {'decay': 0.999,
                           'epsilon': 0.001,
                           'scope': 'batch_norm'}
      with arg_scope([layers.conv2d_transpose],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay'),
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params):

        # project and reshape
        # inputs=random_z: 100 dim
        self.projection = layers.fully_connected(inputs=random_z,
                                                 num_outputs=4*4*512,
                                                 activation_fn=None,
                                                 scope='projection')
        # reshape: 4 x 4 x 512
        self.reshape = tf.reshape(self.projection, [-1, 4, 4, 512], name='reshape')
        # layer1_batch_norm
        self.layer1_batch_norm = layers.batch_norm(self.reshape,
                                                   decay=0.999,
                                                   epsilon=0.001,
                                                   scope='layer1/batch_norm')
        # layer1_relu
        self.layer1_relu = tf.nn.relu(self.layer1_batch_norm, 'layer1_relu')

        # layer2 inputs: 4 x 4 x 512
        self.layer2 = layers.conv2d_transpose(inputs=self.layer1_relu,
                                              num_outputs=256,
                                              scope='layer2')

        # layer3 inputs: 8 x 8 x 256
        self.layer3 = layers.conv2d_transpose(inputs=self.layer2,
                                              num_outputs=128,
                                              scope='layer3')

        # layer4 inputs: 16 x 16 x 128
        self.layer4 = layers.conv2d_transpose(inputs=self.layer3,
                                              num_outputs=64,
                                              scope='layer4')

        # layer5 inputs: 32 x 32 x 64
        self.layer5 = layers.conv2d_transpose(inputs=self.layer4,
                                              num_outputs=3,
                                              activation_fn=tf.tanh,
                                              scope='layer5')

        # output: 64 x 64 x 1
        generated_images = self.layer5

        return generated_images



  def build_random_z_inputs(self):
    # Setup placeholder of random vector z
    with tf.variable_scope('random_z'):
      self.random_z = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.random_z_size],
                                     name='random_z')

    return self.random_z



  def Discriminator(self, images, reuse=False):
    """Discriminator setup.

    Args:
      images: A float32 scalar Tensor of real images from data
      reuse: reuse flag
    """
    with tf.variable_scope('Discriminator') as scope:
      if reuse:
        scope.reuse_variables()

      batch_norm_params = {'decay': 0.999,
                           'epsilon': 0.001,
                           'scope': 'batch_norm'}
      with arg_scope([layers.conv2d],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay')):

        # layer1 inputs: 64 x 64 x 3
        self.layer1 = layers.conv2d(inputs=images,
                                    num_outputs=64,
                                    activation_fn=ops.leakyrelu,
                                    scope='layer1')

        # layer2 inputs: 32 x 32 x 64
        self.layer2 = layers.conv2d(inputs=self.layer1,
                                    num_outputs=128,
                                    normalizer_fn=layers.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=ops.leakyrelu,
                                    scope='layer2')

        # layer3 inputs: 16 x 16 x 128
        self.layer3 = layers.conv2d(inputs=self.layer2,
                                    num_outputs=256,
                                    normalizer_fn=layers.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=ops.leakyrelu,
                                    scope='layer3')
        
        # layer4 inputs: 8 x 8 x 256
        self.layer4 = layers.conv2d(inputs=self.layer3,
                                    num_outputs=512,
                                    normalizer_fn=layers.batch_norm,
                                    normalizer_params=batch_norm_params,
                                    activation_fn=ops.leakyrelu,
                                    scope='layer4')

        # reshape input: 4 x 4 x 512
        self.reshape = tf.reshape(self.layer4, [-1, 4*4*512], name='reshape')

        # layer5 inputs: 4 x 4 x 512
        self.layer5 = layers.fully_connected(inputs=self.reshape,
                                             num_outputs=1,
                                             activation_fn=None,
                                             scope='layer5')

        discriminator_logits = self.layer5

        return discriminator_logits



  def read_real_images_from_placeholder(self):
    # read real images (number character and math symbol dataset)
    # Setup the placeholder of data
    with tf.variable_scope('read_real_images'):
      self.images = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 4096],
                                   name='image')
      real_images = tf.reshape(self.real_images, [-1, 64, 64, 1], name='real_image')

    return real_images


  def read_real_images_from_tfrecords(self):
    # read real images (for celebA)
    with tf.variable_scope('read_real_images'):
      #num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
      num_preprocess_threads = FLAGS.num_preprocess_threads
      real_images = image_processing.distorted_inputs(
                      batch_size=FLAGS.batch_size,
                      num_preprocess_threads=num_preprocess_threads)
    
      #input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    return real_images





  def build(self):
    # generating random vector
    random_z = self.build_random_z_inputs()
    # generating images from Generator() via random vector z
    self.generated_images = self.Generator(random_z)

    # randomly pick up real images from dataset (for number characters and math symbols dataset)
    #self.real_images = self.read_real_images_from_placeholder()
    # randomly pick up real images from dataset (for celebA dataset)
    self.real_images = self.read_real_images_from_tfrecords()

    # discriminating real images by Discriminator()
    self.real_logits = self.Discriminator(self.real_images)
    # discriminating fake images (generated_images) by Discriminator()
    self.fake_logits = self.Discriminator(self.generated_images, reuse=True)

    # losses of real with label "1"
    self.loss_real = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            #labels=tf.ones_like(self.real_logits),
                            # one side label smoothing
                            labels=tf.fill(self.real_logits.get_shape(), 0.9),
                            #labels=tf.ones_like(self.real_logits)*0.9, # the same effect above line
                            logits=self.real_logits))

    # losses of fake with label "0"
    self.loss_fake = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf.zeros_like(self.fake_logits),
                            logits=self.fake_logits))

    # losses of Discriminator
    self.loss_Discriminator = self.loss_real + self.loss_fake

    # losses of Generator with label "1"
    self.loss_Generator = tf.reduce_mean(
                              tf.nn.sigmoid_cross_entropy_with_logits(
                                  labels=tf.ones_like(self.fake_logits),
                                  logits=self.fake_logits))
    
    # Separate variables for each function
    t_vars = tf.trainable_variables()
    
    self.D_vars = [var for var in t_vars if 'Discriminator' in var.name]
    self.G_vars = [var for var in t_vars if 'Generator' in var.name]

    for var in self.G_vars:
      print(var.name)
    for var in self.D_vars:
      print(var.name)

    # Add summaries.
    # Add loss summaries
    tf.summary.scalar("losses/loss_Discriminator", self.loss_Discriminator)
    tf.summary.scalar("losses/loss_Generator", self.loss_Generator)
    tf.summary.scalar("losses/loss_real", self.loss_real)
    tf.summary.scalar("losses/loss_fake", self.loss_fake)

    # Add histogram summaries
    for var in self.D_vars:
      self.D_summary = tf.summary.histogram(var.op.name, var)
    for var in self.G_vars:
      self.G_summary = tf.summary.histogram(var.op.name, var)

    # Add image summaries
    tf.summary.image('random_images', self.generated_images, max_outputs=10)
    tf.summary.image('real_images', self.real_images, max_outputs=10)


    print('complete model build.')


  def visualize_Generator(self):
    # Rescale to [0, 255] instead of [-1, 1]
    generated_images = tf.add(self.generated_images, 1.0)
    #generated_images = tf.multiply(generated_images, 0.5)
    #generated_images = tf.multiply(generated_images, 255.0)
    generated_images = tf.multiply(generated_images, 0.5 * 255.0)
    generated_images = tf.to_int32(generated_images)

    return generated_images




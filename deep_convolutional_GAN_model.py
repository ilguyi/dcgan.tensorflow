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

  def __init__(self, mode):
    """Basic setup.
    """
    assert mode in ["train", "generate"]
    self.mode = mode

    # A int32 scalar value;
    self.random_z_size = 100

    # A int32 scalar value;
    self.batch_size = FLAGS.batch_size

    # A int32 scalar value;
    self.num_preprocess_threads = FLAGS.num_preprocess_threads

    print('The mode is %s.' % self.mode)
    print('complete initializing model.')


  def build_random_z_inputs(self):
    """Build random_z.

    Returns:
      A float32 Tensor with [batch, 1, 1, random_z_size]
    """
    # Setup placeholder of random vector z
    with tf.variable_scope('random_z'):
      if self.mode == "generate":
        self.random_z = tf.placeholder(dtype=tf.float32,
                                       shape=[None, 1, 1, self.random_z_size],
                                       name='random_z')
      else:
        self.random_z = tf.random_uniform([self.batch_size, 1, 1, self.random_z_size],
                                           minval=-1,
                                           maxval=1,
                                           dtype=tf.float32)
    return self.random_z


  def Generator(self, random_z, is_training=True):
    """Generator setup.

    Args:
      random_z: A float32 Tensor random vector (latent code)

    Returns:
      A float32 scalar Tensor of generated images from random vector
    """
    with tf.variable_scope('Generator'):
      batch_norm_params = {'decay': 0.999,
                           'epsilon': 0.001,
                           'is_training': is_training,
                           'scope': 'batch_norm'}
      with arg_scope([layers.conv2d_transpose],
                      kernel_size=[4, 4],
                      stride=[2, 2],
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay')):

        # project and reshape
        # inputs=random_z: 100 dim
        #self.projection = layers.fully_connected(inputs=random_z,
        #                                         num_outputs=4*4*512,
        #                                         activation_fn=None,
        #                                         scope='projection')
        # reshape: 4 x 4 x 512
        #self.reshape = tf.reshape(self.projection, [-1, 4, 4, 512], name='reshape')
        # layer1_batch_norm
        #self.layer1_batch_norm = layers.batch_norm(self.reshape,
        #                                           decay=0.999,
        #                                           epsilon=0.001,
        #                                           scope='layer1/batch_norm')
        # layer1_relu
        #self.layer1_relu = tf.nn.relu(self.layer1_batch_norm, 'layer1_relu')


        # Use full conv2d_transpose instead of project and reshape
        # inputs = random_z: 1 x 1 x 100 dim
        self.layer1 = layers.conv2d_transpose(inputs=random_z,
                                              num_outputs=64 * 8,
                                              padding='VALID',
                                              scope='layer1')
        # layer1: 4 x 4 x (64 * 8)
        self.layer2 = layers.conv2d_transpose(inputs=self.layer1,
                                              num_outputs=64 * 4,
                                              scope='layer2')
        # layer2: 8 x 8 x (64 * 4)
        self.layer3 = layers.conv2d_transpose(inputs=self.layer2,
                                              num_outputs=64 * 2,
                                              scope='layer3')
        # layer3: 16 x 16 x (64 * 2)
        self.layer4 = layers.conv2d_transpose(inputs=self.layer3,
                                              num_outputs=64 * 1,
                                              scope='layer4')
        # layer4: 32 x 32 x (64 * 1)
        self.layer5 = layers.conv2d_transpose(inputs=self.layer4,
                                              num_outputs=3,
                                              normalizer_fn=None,
                                              biases_initializer=None,
                                              activation_fn=tf.tanh,
                                              scope='layer5')
        # output = layer5: 64 x 64 x 3
        generated_images = self.layer5

        return generated_images


  def read_real_images_from_tfrecords(self):
    # read real images (for celebA)
    with tf.variable_scope('read_real_images'):
      #num_preprocess_threads = self.num_preprocess_threads * self.num_gpus
      num_preprocess_threads = self.num_preprocess_threads
      real_images = image_processing.distorted_inputs(
                      batch_size=self.batch_size,
                      num_preprocess_threads=num_preprocess_threads)
      return real_images


  def Discriminator(self, images, reuse=False):
    """Discriminator setup.

    Args:
      images: A float32 scalar Tensor of real images from data
      reuse: reuse flag

    Returns:
      logits: A float32 scalar Tensor
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
                      activation_fn=ops.leakyrelu,
                      normalizer_fn=layers.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=layers.l2_regularizer(0.00004, scope='l2_decay')):

        # images: 64 x 64 x 3
        self.layer1 = layers.conv2d(inputs=images,
                                    num_outputs=64 * 1,
                                    normalizer_fn=None,
                                    biases_initializer=None,
                                    scope='layer1')
        # layer1: 32 x 32 x (64 * 1)
        self.layer2 = layers.conv2d(inputs=self.layer1,
                                    num_outputs=64 * 2,
                                    scope='layer2')
        # layer2: 16 x 16 x (64 * 2)
        self.layer3 = layers.conv2d(inputs=self.layer2,
                                    num_outputs=64 * 4,
                                    scope='layer3')
        # layer3: 8 x 8 x (64 * 4)
        self.layer4 = layers.conv2d(inputs=self.layer3,
                                    num_outputs=64 * 8,
                                    scope='layer4')
        # reshape input: 4 x 4 x 512
        #self.reshape = tf.reshape(self.layer4, [-1, 4*4*512], name='reshape')
        # layer5 inputs: 4 x 4 x 512
        #self.layer5 = layers.fully_connected(inputs=self.reshape,
        #                                     num_outputs=1,
        #                                     activation_fn=None,
        #                                     scope='layer5')
        self.layer5 = layers.conv2d(inputs=self.layer4,
                                    num_outputs=1,
                                    stride=[1, 1],
                                    padding='VALID',
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    activation_fn=None,
                                    scope='layer5')

        discriminator_logits = self.layer5

        return discriminator_logits



  def build(self):
    # generating random vector
    random_z = self.build_random_z_inputs()
    
    if self.mode == "generate":
      # generating images from Generator() via random vector z
      self.generated_images = self.Generator(random_z, is_training=False)

    if self.mode == "train":
      # generating images from Generator() via random vector z
      self.generated_images = self.Generator(random_z)

      # randomly pick up real images from dataset (for celebA dataset)
      self.real_images = self.read_real_images_from_tfrecords()

      # discriminating real images by Discriminator()
      self.real_logits = self.Discriminator(self.real_images)

      # discriminating fake images (generated_images) by Discriminator()
      self.fake_logits = self.Discriminator(self.generated_images, reuse=True)

      # losses of real with label "1"
      self.loss_real = ops.GANLoss(logits=self.real_logits, is_real=True)
      # losses of fake with label "0"
      self.loss_fake = ops.GANLoss(logits=self.fake_logits, is_real=False)

      # losses of Discriminator
      self.loss_Discriminator = self.loss_real + self.loss_fake

      # losses of Generator with label "1"
      self.loss_Generator = ops.GANLoss(logits=self.fake_logits, is_real=True)

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
        tf.summary.histogram(var.op.name, var)
      for var in self.G_vars:
        tf.summary.histogram(var.op.name, var)

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




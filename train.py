from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import copy

import tensorflow as tf

import deep_convolutional_GAN_model as dcgan

slim = tf.contrib.slim


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
tf.app.flags.DEFINE_float('adam_beta1',
                          0.9,
                          'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('adam_beta2',
                          0.999,
                          'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('adam_epsilon',
                          1e-08,
                          'Epsilon term for the optimizer.')

FLAGS = tf.app.flags.FLAGS



def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if tf.gfile.Exists(FLAGS.train_dir):
    raise ValueError('This folder already exists.')
  tf.gfile.MakeDirs(FLAGS.train_dir)


  with tf.Graph().as_default():

    # Build the model.
    model = dcgan.DeepConvGANModel(mode="train")
    model.build()

    # Create global step
    global_step = slim.create_global_step()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (FLAGS.num_examples / FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.train.exponential_decay(
          FLAGS.initial_learning_rate,
          global_step,
          decay_steps=decay_steps,
          decay_rate=FLAGS.learning_rate_decay_factor,
          staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Create an optimizer that performs gradient descent for Discriminator.
    opt_D = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.adam_epsilon)

    # Create an optimizer that performs gradient descent for Discriminator.
    opt_G = tf.train.AdamOptimizer(
                learning_rate,
                beta1=FLAGS.adam_beta1,
                beta2=FLAGS.adam_beta2,
                epsilon=FLAGS.adam_epsilon)

    # Minimize optimizer
    opt_op_D = opt_D.minimize(model.loss_Discriminator,
                              global_step=global_step,
                              var_list=model.D_vars)
    opt_op_G = opt_G.minimize(model.loss_Generator,
                              global_step=global_step,
                              var_list=model.G_vars)

    # Compute the gradients for a list of variables.
    # grads_and_vars is a list of tuples (gradients, variables).
#    grads_and_vars = opt.compute_gradients(model.total_loss)
#
#    # Apply the gradients
#    apply_gradient_op = opt.apply_gradients(grads_and_vars)


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=1000)


    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session()
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)


    # Create a summary writer, add the 'graph' to the event file.
    summary_writer = tf.summary.FileWriter(
                        FLAGS.train_dir,
                        sess.graph)

    # Retain the summaries
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

    # Build the summary operation
    summary_op = tf.summary.merge(summaries)


    for step in range(FLAGS.max_steps+1):
      start_time = time.time()
      loss_D, loss_G, _, _ = sess.run([model.loss_Discriminator,
                                       model.loss_Generator,
                                       opt_op_D, opt_op_G])

      epochs = step*FLAGS.batch_size/FLAGS.num_examples
      #if epochs < 1:
      #  sess.run([opt_op_D])

      duration = time.time() - start_time

      if step % 10 == 0:
        examples_per_sec = FLAGS.batch_size / float(duration)
        print("Epochs: %.2f step: %d  loss_D: %f loss_G: %f (%.1f examples/sec; %.3f sec/batch)"
                % (epochs, step, loss_D, loss_G, examples_per_sec, duration))
        
      if step % 200 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % FLAGS.save_steps == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

    print('complete training...')



if __name__ == '__main__':
  tf.app.run()

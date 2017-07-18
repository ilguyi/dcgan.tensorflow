from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import tensorflow as tf

import configuration
import deep_convolutional_GAN_model as dcgan

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS


def SetOptimizer(learning_rate, optimizer):
  if FLAGS.optimizer == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(learning_rate)
  elif FLAGS.optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(learning_rate)
  elif FLAGS.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate,
                                 beta1=FLAGS.adam_beta1,
                                 beta2=FLAGS.adam_beta2,
                                 epsilon=FLAGS.adam_epsilon)
  elif FLAGS.optimizer == 'ftrl':
    opt = tf.train.FtrlOptimizer(learning_rate)
  elif FLAGS.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(learning_rate)
  elif FLAGS.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
  elif FLAGS.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)

  return opt



def main(_):

  # train_dir path in each the combination of hyper-parameters
  train_dir = configuration.hyperparameters_dir(FLAGS.train_dir)

  if tf.gfile.Exists(train_dir):
    raise ValueError('This folder already exists.')
  tf.gfile.MakeDirs(train_dir)

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
    opt_D = SetOptimizer(learning_rate, FLAGS.optimizer)
    # Create an optimizer that performs gradient descent for Generator.
    opt_G = SetOptimizer(learning_rate, FLAGS.optimizer)

    # Minimize optimizer
    opt_op_D = opt_D.minimize(model.loss_Discriminator,
                              global_step=global_step,
                              var_list=model.D_vars)
    opt_op_G = opt_G.minimize(model.loss_Generator,
                              global_step=global_step,
                              var_list=model.G_vars)

    # Track the moving averages of all trainable variables.
    # Note that we maintain a "double-average" of the BatchNormalization
    # global statistics. This is more complicated then need be but we employ
    # this for backward-compatibility with our previous models.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, global_step)

    # Another possibility is to use tf.slim.get_variables().
    variables_to_average = (tf.trainable_variables() +
                            tf.moving_average_variables())
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Batch normalization update
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

    train_op = tf.group(opt_op_D, opt_op_G, variables_averages_op,
                        batchnorm_updates_op)

    # Add dependency to compute batchnorm_updates.
    with tf.control_dependencies([variables_averages_op, batchnorm_updates_op]):
      opt_op_D
      opt_op_G

    # Compute the gradients for a list of variables.
    # grads_and_vars is a list of tuples (gradients, variables).
#    grads_and_vars = opt.compute_gradients(model.total_loss)
#
#    # Apply the gradients
#    apply_gradient_op = opt.apply_gradients(grads_and_vars)


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)


    # Start running operations on the Graph.
    with tf.Session() as sess:
      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      # Create a summary writer, add the 'graph' to the event file.
      summary_writer = tf.summary.FileWriter(
                          train_dir,
                          sess.graph)

      # Retain the summaries
      #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
      # Build the summary operation
      #summary_op = tf.summary.merge(summaries)

      # Retain the summaries and Build the summary operation
      summary_op = tf.summary.merge_all()

      for step in range(FLAGS.max_steps+1):
        start_time = time.time()
        _, loss_D, loss_G = sess.run([train_op,
                                      model.loss_Discriminator,
                                      model.loss_Generator])

        epochs = step * FLAGS.batch_size / FLAGS.num_examples
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
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    print('complete training...')



if __name__ == '__main__':
  tf.app.run()

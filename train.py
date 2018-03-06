from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time

import tensorflow as tf

import configuration
import deep_convolutional_GAN_model as dcgan


FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


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

  with tf.Graph().as_default():

    # Build the model.
    model = dcgan.DeepConvGANModel(mode="train")
    model.build()

    # Create global step
    #global_step = tf.train.create_global_step()

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (FLAGS.num_examples / FLAGS.batch_size)
    print("num_batches_per_epoch: %f" % num_batches_per_epoch)
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

    # Decay the learning rate exponentially based on the number of steps.
    learning_rate = tf.constant(FLAGS.initial_learning_rate)
    def _learning_rate_decay_fn(learning_rate, global_step):
      return tf.train.exponential_decay(
                learning_rate,
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
    # one training step is defined by both optimizers run once.
#    opt_D_op = opt_D.minimize(model.loss_Discriminator,
#                              var_list=model.D_vars)
#    opt_G_op = opt_G.minimize(model.loss_Generator,
#                              global_step=model.global_step,
#                              var_list=model.G_vars)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.MOVING_AVERAGE_DECAY, model.global_step)
    variables_to_average = tf.trainable_variables()
    variables_averages_op = variable_averages.apply(variables_to_average)

    # Batch normalization update
    batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    batchnorm_updates_op = tf.group(*batchnorm_updates)

#    train_op = tf.group(opt_D_op, opt_G_op, variables_averages_op,
#                        batchnorm_updates_op)

    # Add dependency to compute batchnorm_updates.
    with tf.control_dependencies([variables_averages_op, batchnorm_updates_op]):
      # Minimize optimizer
      opt_D_op = opt_D.minimize(model.loss_Discriminator,
                                var_list=model.D_vars)
      opt_G_op = opt_G.minimize(model.loss_Generator,
                                global_step=model.global_step,
                                var_list=model.G_vars)


    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

    # Build the summary operation
    summary_op = tf.summary.merge_all()

    # train_dir path in each the combination of hyper-parameters
    train_dir = configuration.hyperparameters_dir(FLAGS.train_dir)

    # Training with tf.train.Supervisor.
    sv = tf.train.Supervisor(logdir=train_dir,
                             summary_op=None,     # Do not run the summary services
                             saver=saver,
                             save_model_secs=0,   # Do not run the save_model services
                             init_fn=None)        # Not use pre-trained model
    # Start running operations on the Graph.
    with sv.managed_session() as sess:
      tf.logging.info('Start Session.')

      # Start the queue runners.
      sv.start_queue_runners(sess=sess)
      tf.logging.info('Starting Queues.')

      # Run a model
      for epoch in range(FLAGS.max_epochs):
        for j in range(int(num_batches_per_epoch)):
          start_time = time.time()
          if sv.should_stop():
            break

          for _ in range(FLAGS.k):
            _, loss_D = sess.run([opt_D_op, model.loss_Discriminator])
          _, _global_step, loss_G = sess.run([opt_G_op,
                                              sv.global_step,
                                              model.loss_Generator])

          epochs = epoch + j / num_batches_per_epoch
          duration = time.time() - start_time

          # Monitoring training situation in console.
          if _global_step % 10 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            print("Epochs: %.3f global step: %d  loss_D: %f loss_G: %f (%.1f examples/sec; %.3f sec/batch)"
                    % (epochs, _global_step, loss_D, loss_G, examples_per_sec, duration))
            
          # Save the model summaries periodically.
          if _global_step % 200 == 0:
            summary_str = sess.run(summary_op)
            sv.summary_computed(sess, summary_str)

          # Save the model checkpoint periodically.
          if epoch % FLAGS.save_epochs == 0  and  j == 0:
            tf.logging.info('Saving model with global step %d (= %d epoch) to disk.' % (_global_step, epoch))
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

    tf.logging.info('complete training...')



if __name__ == '__main__':
  tf.app.run()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import copy
import cv2

import tensorflow as tf

import image_processing
import deep_convolutional_GAN_model


slim = tf.contrib.slim


####################
# Generating Flags #
####################
tf.app.flags.DEFINE_string('checkpoint_path',
                           '',
                           'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('checkpoint_step',
                            -1,
                            'The step you want to read model checkpoints.'
                            '-1 means the latest model checkpoints.')
tf.app.flags.DEFINE_integer('batch_size',
                            32,
                            'The number of samples in each batch.')
tf.app.flags.DEFINE_string('dataset_dir',
                           None,
                           'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('seed',
                            0,
                            'The seed number.')
tf.app.flags.DEFINE_boolean('make_gif',
                            False,
                            'Whether make gif or not.')
tf.app.flags.DEFINE_integer('save_steps',
                            5000,
                            'The step per saving model.')
tf.app.flags.DEFINE_integer('column_index',
                            0,
                            'The column index of random vector for linear interpolation.')

FLAGS = tf.app.flags.FLAGS



def make_squared_image(generated_images):
  N = len(generated_images)
  black_image = np.zeros(generated_images[0].shape, dtype=np.int32)
  w = int(np.minimum(10, np.sqrt(N)))
  h = int(np.ceil(N / w))

  one_row_image = generated_images[0]
  for j in range(1, w):
    one_row_image = np.concatenate((one_row_image, generated_images[j]), axis=1)
  
  image = one_row_image
  for i in range(1, h):
    one_row_image = generated_images[i*w]
    for j in range(1, w):
      try:
        one_row_image = np.concatenate((one_row_image, generated_images[i*w + j]), axis=1)
      except:
        one_row_image = np.concatenate((one_row_image, black_image), axis=1)
    image = np.concatenate((image, one_row_image), axis=0)

  return image



def ImageWrite(image, step):
  r,g,b = cv2.split(image)
  image = cv2.merge([b,g,r])

  filename = 'generated_images_%06.d.jpg' % step
  cv2.imwrite(filename, image)



def GIFWrite(generated_gifs, duration=4):
  # below code does not working
  # convert -delay 20 -loop 0 *.jpg generated_images.gif
  # after make generated_images in each step
#  import moviepy.editor as mpy
#
#  def make_frame(t):
#    try:
#      x = generated_gifs[int(len(generated_gifs)*t/duration)]
#    except:
#      x = generated_gifs[-1]
#
#    return x
#
#  filename = 'generated_images.gif'
#  clip = mpy.VideoClip(make_frame, duration=duration)
#  clip.write_gif(filename, fps=len(generated_gifs)/duration)

  for i, image in enumerate(generated_gifs):
    ImageWrite(image, i*FLAGS.save_steps)
  



def run_generator_once(saver, checkpoint_path, model, random_z):
  print(checkpoint_path)
  start_time = time.time()
  with tf.Session() as sess:
    tf.logging.info("Loading model from checkpoint: %s", checkpoint_path)
    saver.restore(sess, checkpoint_path)
    tf.logging.info("Successfully loaded checkpoint: %s",
                    os.path.basename(checkpoint_path))


    generated_images = model.visualize_Generator()
    feed_dict={model.random_z: random_z}
    generated_images = sess.run(generated_images,
                               feed_dict=feed_dict)

    duration = time.time() - start_time
    print("Loading time: %.3f" % duration)

  return generated_images




def main(_):
  if not FLAGS.checkpoint_path:
    raise ValueError('You must supply the checkpoint_path with --checkpoint_path')


  with tf.Graph().as_default():
    start_time = time.time()

    # Build the generative model.
    model = deep_convolutional_GAN_model.DeepConvGANModel()
    model.build()

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver()

    if not FLAGS.make_gif:
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        if FLAGS.checkpoint_step == -1:
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
          checkpoint_step = int(checkpoint_path.split('-')[1])
        else:
          checkpoint_path = os.path.join(FLAGS.checkpoint_path, 'model.ckpt-%d' % FLAGS.checkpoint_step)
          checkpoint_step = FLAGS.checkpoint_step

        if os.path.basename(checkpoint_path) + '.data-00000-of-00001' in os.listdir(FLAGS.checkpoint_path):
          print(os.path.basename(checkpoint_path))
        else:
          raise ValueError("No checkpoint file found in: %s" % checkpoint_path)
      else:
        raise ValueError("checkpoint_path must be folder path")


      # Set fixed random vectors
      np.random.seed(FLAGS.seed)
      random_z = np.random.uniform(-1, 1, [FLAGS.batch_size, 100])

      # Set random vector for linear interpolation
      #random_z_one = np.random.uniform(-1, 1, [1, 100])
      #random_z = random_z_one
      #for i in range(FLAGS.batch_size-1):
      #  random_z = np.concatenate((random_z, random_z_one), axis=0)

      #linear_interpolation = np.linspace(-1.0, 1.0, num=FLAGS.batch_size)
      #random_z[:, FLAGS.column_index] = linear_interpolation

      generated_images = run_generator_once(saver, checkpoint_path, model, random_z)
      squared_images = make_squared_image(generated_images)

      ImageWrite(squared_images, checkpoint_step)

    else:
      # Find all checkpoint_path
      if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_filenames = []
        for filename in os.listdir(FLAGS.checkpoint_path):
          if '.data-00000-of-00001' in filename:
            filename = filename.split(".")[1].split("ckpt-")[1]
            checkpoint_filenames.append(filename)
      else:
        raise ValueError("checkpoint_path must be folder path")

      checkpoint_filenames.sort(key=int)
      for i, filename in enumerate(checkpoint_filenames):
        filename = 'model.ckpt-' + filename
        checkpoint_filenames[i] = filename

      # Set fixed random vectors
      np.random.seed(FLAGS.seed)
      random_z = np.random.uniform(-1, 1, [FLAGS.batch_size, 100])

      generated_gifs = []
      for checkpoint_path in checkpoint_filenames:
        checkpoint_path = os.path.join(FLAGS.checkpoint_path, checkpoint_path)
        generated_images = run_generator_once(saver, checkpoint_path, model, random_z)
        squared_images = make_squared_image(generated_images)
        generated_gifs.append(squared_images)

      GIFWrite(generated_gifs)


    print('complete generating image...')




if __name__ == '__main__':
  tf.app.run()

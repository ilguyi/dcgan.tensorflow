from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import cv2

import tensorflow as tf

import configuration
import deep_convolutional_GAN_model as dcgan

slim = tf.contrib.slim

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

  for image, step in generated_gifs:
    ImageWrite(image, step)
  



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
  if not FLAGS.checkpoint_dir:
    raise ValueError('You must supply the checkpoint_dir with --checkpoint_dir')

  # checkpoint_dir in each the combination of hyper-parameters
  checkpoint_dir = configuration.hyperparameters_dir(FLAGS.checkpoint_dir)

  if not tf.gfile.IsDirectory(checkpoint_dir):
    raise ValueError('checkpoint_dir must be folder path')

  with tf.Graph().as_default():
    # Build the generative model.
    model = dcgan.DeepConvGANModel(mode="generate")
    model.build()
    
    # Restore the moving average version of the learned variables for image translate.
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(variables_to_restore)

    if not tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
      raise ValueError("checkpoint_dir must be folder path")

    # Generate images for all checkpoints
    if FLAGS.make_gif:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

      # Set fixed random vectors
      np.random.seed(FLAGS.seed)
      random_z = np.random.uniform(-1, 1, [FLAGS.batch_size, 1, 1, 100])

      generated_gifs = []
      for checkpoint_path in ckpt.all_model_checkpoint_paths:
        if not os.path.exists(os.path.join(checkpoint_path + '.data-00000-of-00001')):
          raise ValueError("No checkpoint files found in: %s" % checkpoint_path)
        print(checkpoint_path)

        generated_images = run_generator_once(saver, checkpoint_path, model, random_z)
        squared_images = make_squared_image(generated_images)
        checkpoint_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
        generated_gifs.append((squared_images, checkpoint_step))

      GIFWrite(generated_gifs)

    else:
      if FLAGS.checkpoint_step == -1:
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
      else:
        checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt-%d' % FLAGS.checkpoint_step)

      if not os.path.exists(os.path.join(checkpoint_path + '.data-00000-of-00001')):
        raise ValueError("No checkpoint file found in: %s" % checkpoint_path)

      # Set fixed random vectors
      np.random.seed(FLAGS.seed)
      random_z = np.random.uniform(-1, 1, [FLAGS.batch_size, 1, 1, 100])

      generated_images = run_generator_once(saver, checkpoint_path, model, random_z)
      squared_images = make_squared_image(generated_images)

      checkpoint_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
      ImageWrite(squared_images, checkpoint_step)


    print('complete generating image...')




if __name__ == '__main__':
  tf.app.run()

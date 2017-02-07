from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.framework import ops

import tensorflow as tf

layers = tf.contrib.layers





def leakyrelu(x, leaky_weight=0.2, name=None):
  """Computes leaky relu of `x` element-wise.

  Args:
    x: A Tensor with type `float`, `double`, `int32`,
      `complex64`, `int64`, or `qint32`.
    name: A name for the operation (optional).

  Returns:
    A Tensor respectively with the same type as `x` if
    `x.dtype != qint32` otherwise the return type is `qint8`.
  """
  with ops.name_scope(name, "LRelu", [x]) as name:
    return tf.maximum(x, leaky_weight*x)


# deprecated
def generator_layer(inputs, num_outputs,
                    activation_fn='relu',
                    scope=None):
  """Compute conv2d_transpose -> batch_norm -> activation in order
     for Generator.

  Args:
    inputs: A Tensor
    num_outputs: number of outputs
    activation_fn: 'relu' or 'tanh'
    scope: scope

  Returns:
    A Tensor
  """
  with tf.variable_scope(scope):
    # conv2d_transpose
    conv2d_t = layers.conv2d_transpose(inputs=inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=[4, 4],
                                       stride=[2, 2],
                                       activation_fn=None,
                                       scope='conv2d_t')

    # batch_norm
    batch_norm = layers.batch_norm(inputs=conv2d_t,
                                   decay=0.999,
                                   epsilon=0.001,
                                   scope='batch_norm')

    # activation_fn
    if activation_fn == 'relu':
      activation = tf.nn.relu(batch_norm, 'relu')
    elif activation_fn == 'tanh':
      activation = tf.tanh(batch_norm, 'tanh')
    else:
      raise ValueError('activation_fn must be \'relu\' or \'tanh\'')

    return activation
  


# deprecated
def discriminator_layer(inputs, num_outputs,
                        batch_norm_flag=True,
                        activation_fn='leakyrelu',
                        scope=None):
  """Compute conv2d -> batch_norm -> activation in order
     for Discriminator.

  Args:
    inputs: A Tensor
    num_outputs: number of outputs
    batch_norm_flag: whether use batch_norm layer (True or False)
    activation_fn: 'leakyrelu' or 'sigmoid'
    scope: scope

  Returns:
    A Tensor
  """
  with tf.variable_scope(scope):
    # conv2d
    conv2d = layers.conv2d(inputs=inputs,
                           num_outputs=num_outputs,
                           kernel_size=[4, 4],
                           stride=[2, 2],
                           activation_fn=None,
                           scope='conv2d')

    # batch_norm
    if batch_norm_flag:
      batch_norm = layers.batch_norm(inputs=conv2d,
                                     decay=0.999,
                                     epsilon=0.001,
                                     scope='batch_norm')
    else:
      batch_norm = conv2d

    # activation_fn
    if activation_fn == 'leakyrelu':
      activation = leakyrelu(batch_norm, leaky_weight=0.2, name='leakyrelu')
    elif activation_fn == 'sigmoid':
      activation = tf.sigmoid(batch_norm, 'sigmoid')
    else:
      raise ValueError('activation_fn must be \'leakyrelu\' or \'sigmoid\'')

    return activation
  



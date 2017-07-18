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


def GANLoss(logits, is_real=True, smoothing=0.9, name=None):
  """Computes standard GAN loss between `logits` and `labels`.

  Args:
    logits: A float32 Tensor of logits.
    is_real: boolean, True means `1` labeling, False means `0` labeling.
    smoothing: one side labels smoothing.

  Returns:
    A scalar Tensor representing the loss value.
  """
  if is_real:
    # one side label smoothing
    labels = tf.fill(logits.get_shape(), smoothing)
  else:
    labels = tf.zeros_like(logits)

  with ops.name_scope(name, 'GAN_loss', [logits, labels]) as name:
    loss = tf.reduce_mean(
              tf.nn.sigmoid_cross_entropy_with_logits(
                              labels=labels,
                              logits=logits))
    return loss



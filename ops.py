from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

import tensorflow as tf


def GANLoss(logits, is_real=True, smoothing=1.0, scope=None):
  """Computes standard GAN loss between `logits` and `labels`.

  Args:
    logits: A float32 Tensor of logits.
    is_real: boolean, True means `1` labeling, False means `0` labeling.
    smoothing: one side labels smoothing.
    scope: name scope.

  Returns:
    A scalar Tensor representing the loss value.
  """
  if is_real:
    labels = tf.ones_like(logits)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                           logits=logits,
                                           label_smoothing=smoothing,
                                           scope=scope)
  else:
    labels = tf.zeros_like(logits)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,
                                           logits=logits,
                                           scope=scope)
  return loss



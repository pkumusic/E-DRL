"""Loss functions."""

import tensorflow as tf
import semver
import numpy as np


def huber_loss(y_true, y_pred, max_grad=1.):
    """Calculate the huber loss.

    See https://en.wikipedia.org/wiki/Huber_loss

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The huber loss.
    """
    x = y_true - y_pred
    sqrcost = tf.square(x)
    abscost = tf.abs(x)
    return tf.where(abscost < max_grad,
              sqrcost * 0.5,
              abscost * max_grad - 0.5 * max_grad ** 2)


def mean_huber_loss(y_true, y_pred, max_grad=1., name='huber_loss'):
    """Return mean huber loss.

    Same as huber_loss, but takes the mean over all values in the
    output tensor.

    Parameters
    ----------
    y_true: np.array, tf.Tensor
      Target value.
    y_pred: np.array, tf.Tensor
      Predicted value.
    max_grad: float, optional
      Positive floating point value. Represents the maximum possible
      gradient magnitude.

    Returns
    -------
    tf.Tensor
      The mean huber loss.
    """

    return tf.reduce_mean(huber_loss(y_true, y_pred, max_grad), name=name)

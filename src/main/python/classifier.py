
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import vgg19_trainable as vgg

# built from https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist.py


def inference(images, label_count, weights1, weights2):
  net = vgg.Vgg19(weights1=weights1, weights2=weights2)
  net.build(images)
  
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([weights2, label_count],
                            stddev=1.0 / math.sqrt(float(weights2))),
                            name='weights')
    biases = tf.Variable(tf.zeros([label_count]), name='biases')
    logits = tf.nn.relu(tf.matmul(net.fc7, weights) + biases)
  return logits


def loss(logits, labels, false_positive_weight=.5):
  """Calculates the loss from the logits and the labels."""
  with tf.name_scope('loss'):
    # NOTE: We probably want to treat false negatives worse than false positives
    # If we think something is there that isn't, it's less bad than if we think something isn't
    # there that should be. This is because the expected tag space will be pretty big, and
    # images will generally have a relatively small fraction of enabled tags
    
    false_positives = tf.reduce_sum(tf.maximum(labels-logits, 0))
    false_negatives = tf.reduce_sum(tf.maximum(logits-labels, 0))
    
    return false_positives * false_positive_weight + false_negatives


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  """Return the number of correct predictions
  Args:
    logits: Logits tensor, float - [batch_size, tags_size]
    labels: Labels tensor, float - [batch_size, tags_size]
  """
  
  positives = labels
  negatives = 1-labels

  false_positives = tf.maximum(labels-logits, 0)
  false_negatives = tf.maximum(logits-labels, 0)
  
  # Later, we should indicate false positives and false negatives
  return positives, negatives, false_positives, false_negatives

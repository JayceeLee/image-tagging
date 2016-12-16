
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
  
  with tf.name_scope('logits'):
    weights = tf.Variable(
        tf.truncated_normal([weights2, label_count],
                            stddev=1.0 / math.sqrt(float(weights2))),
                            name='weights')
    biases = tf.Variable(tf.zeros([label_count]), name='biases')
    linear = tf.matmul(net.fc7, weights) + biases
    logits = tf.sigmoid(tf.nn.dropout(linear, 0.5))
  return logits


def loss(logits, labels, tags_to_evaluate, false_negative_weight=2.0):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, tags_size]
    labels: Labels tensor, float - [batch_size, tags_size]
  """
  with tf.name_scope('loss'):
    # NOTE: We probably want to treat false negatives worse than false positives
    # If we think something is there that isn't, it's less bad than if we think something isn't
    # there that should be. This is because the expected tag space will be pretty big, and
    # images will generally have a relatively small fraction of enabled tags
    
    sliced_logits = tf.slice(logits, [0,0], [-1,tags_to_evaluate])
    sliced_labels = tf.slice(labels, [0,0], [-1,tags_to_evaluate])

#    return tf.contrib.losses.absolute_difference(sliced_logits, sliced_labels)

    false_positives = tf.maximum(sliced_labels-sliced_logits, 0.0)
    false_negatives = tf.maximum(sliced_logits-sliced_labels, 0.0)

    return (tf.contrib.losses.compute_weighted_loss(false_positives)
        + false_negative_weight * tf.contrib.losses.compute_weighted_loss(false_negatives))


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


def evaluation(logits, labels, tags_to_evaluate):
  """Return the number of correct predictions
  Args:
    logits: Logits tensor, float - [batch_size, tags_size]
    labels: Labels tensor, float - [batch_size, tags_size]
  """
  
  sliced_logits = tf.slice(logits, [0,0], [-1,tags_to_evaluate])
  sliced_labels = tf.slice(labels, [0,0], [-1,tags_to_evaluate])
  
  positives = sliced_labels
  negatives = 1-sliced_labels

  false_positives = tf.maximum(sliced_labels-sliced_logits, 0)
  false_negatives = tf.maximum(sliced_logits-sliced_labels, 0)
  
  # Later, we should indicate false positives and false negatives
  return positives, negatives, false_positives, false_negatives

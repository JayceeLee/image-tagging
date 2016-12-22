
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

import dataset
import classifier

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('weights1', 100, 'Number of neurons in first fully connected layer.')
flags.DEFINE_integer('weights2', 100, 'Number of neurons in second fully connected layer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '../../../output/', 'Directory to put the training data.')
flags.DEFINE_integer('summary_interval', 1, 'How often to print summaries')
flags.DEFINE_integer('checkpoint_interval', 20, 'How often to save checkpoints')
flags.DEFINE_float('increment_tags_threshold', 0.25, 'When loss is this low, increment tags to learn')
flags.DEFINE_float('false_negative_weight', 10.0, 'How strongly to weight false negatives (vs false positives)')


def placeholder_inputs(batch_size, images_shape, labels_shape):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  
  images_shape_list = list(images_shape)
  labels_shape_list = list(labels_shape)
  
  images_shape_list[0] = batch_size
  labels_shape_list[0] = batch_size
  
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=tuple(images_shape_list), name="images")
  labels_placeholder = tf.placeholder(tf.float32, shape=tuple(labels_shape_list), name="labels")
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, train_mode_pl, train_mode):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      train_mode_pl: train_mode,
  }
  return feed_dict


def do_eval(sess,
            indices_to_tags,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            train_mode,
            data_set,
            tags_to_evaluate):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  
  with sess.as_default():
    indices_to_tags = {k: v for k, v in indices_to_tags.items() if k < tags_to_evaluate.eval()}
  
  # And run one epoch of eval.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size

  all_positives = np.zeros(len(indices_to_tags), dtype=np.float32)
  all_negatives = np.zeros(len(indices_to_tags), dtype=np.float32)
  all_false_positives = np.zeros(len(indices_to_tags), dtype=np.float32)
  all_false_negatives = np.zeros(len(indices_to_tags), dtype=np.float32)

  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               train_mode, False)
    positives, negatives, false_positives, false_negatives = sess.run(eval_correct, feed_dict=feed_dict)
    all_positives += np.sum(positives,0)
    all_negatives += np.sum(negatives,0)
    all_false_positives += np.sum(false_positives,0)
    all_false_negatives += np.sum(false_negatives,0)
  
  print('  Total correct positives %.2f, out of %.2f' % 
      (np.sum(all_positives) - np.sum(all_false_positives),
      np.sum(all_positives)))
  print('  Total correct negatives %.2f, out of %.2f' % 
      (np.sum(all_negatives) - np.sum(all_false_negatives),
      np.sum(all_negatives)))

  for index in indices_to_tags:
    print('  %s:\t+: %.2f/%.2f\t-: %.2f/%.2f' %
        (indices_to_tags[index],
        all_positives[index] - all_false_positives[index],
        all_positives[index],
        all_negatives[index] - all_false_negatives[index],
        all_negatives[index],))


def run_training():
  """Train MNIST for a number of steps."""
  # Get the sets of images and labels for training, validation, and test
  print('Loading data...')
  train_data, validation_data, indices_to_tags = dataset.load_data(FLAGS.train_dir)
  print('train data: %d' % len(train_data.images))
  print('validation data: %d' % len(validation_data.images))
  print('Data loaded, starting training')

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size, train_data.images.shape, train_data.labels.shape)
    train_mode = tf.placeholder(tf.bool, name="train_mode")

    # Build a Graph that computes predictions from the inference model.
    logits = classifier.inference(
        images_placeholder, len(indices_to_tags), FLAGS.weights1, FLAGS.weights2, train_mode)
        
    # Represents the number of tags to attempt to learn.
    # We will learn tags incrementally
    tags_to_evaluate = tf.Variable(1, name="tags_to_evaluate", trainable=False)
    with tf.name_scope("increment_tags_to_evaluate"):
      increment_tags_to_evaluate = tf.assign(
          tags_to_evaluate, tf.minimum(tf.add(tags_to_evaluate, tf.constant(1)), len(indices_to_tags)))

    # Add to the Graph the Ops for loss calculation.
    loss = classifier.loss(logits, labels_placeholder, tags_to_evaluate, FLAGS.false_negative_weight)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = classifier.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = classifier.evaluation(logits, labels_placeholder, tags_to_evaluate)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.merge_all_summaries()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(train_data,
                                 images_placeholder,
                                 labels_placeholder,
                                 train_mode, True)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % FLAGS.summary_interval == 0:
        # Print status to stdout.
        print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
        print('Saving checkpoint.')
        checkpoint_file = os.path.join(FLAGS.train_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                indices_to_tags,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                train_mode,
                validation_data,
                tags_to_evaluate)

      eval_tags_to_evaluate = sess.run(tags_to_evaluate)
      if loss_value < FLAGS.increment_tags_threshold and eval_tags_to_evaluate < len(indices_to_tags):
        print('Incrementing tags_to_evaluate')
        sess.run(increment_tags_to_evaluate)
        pass


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()

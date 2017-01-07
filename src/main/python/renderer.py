
""" Attempt to construct an image based on an input classification """

import tensorflow as tf
import numpy as np
import os
import math
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy.misc

import dataset

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('category', 123, 'Category to match')
flags.DEFINE_float('learning_rate', .1, 'Learning rate')
flags.DEFINE_string('output_dir', './output', 'Directory to output stuff.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('summary_interval', 1, 'How often to print summaries')
flags.DEFINE_integer('checkpoint_interval', 10, 'How often to save checkpoints')
flags.DEFINE_string('base_name', 'image', 'Base name of the image')
flags.DEFINE_string('base_image', '', 'Image to use as basis')


import vgg19 as vgg

def save_image(filename, data):
  scipy.misc.imsave('%s/%s' % (FLAGS.output_dir, filename), data)

def main(_):
  with tf.Graph().as_default():
    if FLAGS.base_image:
      base_image = dataset.imread(FLAGS.base_image)
      image = tf.Variable(np.array([base_image], dtype = np.float32), name='image')
    else:
      image = tf.Variable(tf.truncated_normal([1, 224, 224, 3], stddev=1.0 / math.sqrt(float(224*224))),
                        name='image')

    tf.image_summary('image', image)

    with tf.name_scope('vgg'):
      net = vgg.Vgg19()
      net.build(image)
      logits = net.fc8
    labels = np.array([FLAGS.category])
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    summary = tf.merge_all_summaries()
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(FLAGS.output_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.initialize_all_variables())
    
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      
      if step % FLAGS.summary_interval == 0:
        _, loss_value, summary_str = sess.run([train_op, loss, summary])
        # Update the events file.
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      else:
        _, loss_value = sess.run([train_op, loss])

      duration = time.time() - start_time
      print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
        eval_image, = sess.run([image])
        save_image('%s-%d-%d.png' % (FLAGS.base_name ,FLAGS.category, step), eval_image[0])
        
        print('Saving checkpoint.')
        checkpoint_file = os.path.join(FLAGS.output_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=step)

if __name__ == '__main__':
  tf.app.run()

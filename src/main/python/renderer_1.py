
""" Attempt to construct an image based on an input classification """

import tensorflow as tf
import numpy as np
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy.misc

import dataset

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('layer', 'conv4_2', 'Layer to use')
flags.DEFINE_integer('channel', 123, 'Channel to match')
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
      image = np.array([base_image], dtype = np.float32)
    else:
      image = np.random.uniform(size=(1,224,224,3)).astype(np.float32) + 100.0

    image_placeholder = tf.placeholder(tf.float32, shape=image.shape, name='image_pl')
    tf.image_summary('image', image_placeholder)

    with tf.name_scope('vgg'):
      net = vgg.Vgg19()
      net.build(image_placeholder)

    loss = tf.reduce_mean(getattr(net, FLAGS.layer)[:,:,:,FLAGS.channel])
    loss_grad = tf.gradients(loss, image_placeholder)[0]
    
    tf.scalar_summary(loss.op.name, loss)
    
    summary = tf.merge_all_summaries()
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(FLAGS.output_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      
      loss_grad_value, loss_value, summary_str = sess.run([loss_grad, loss, summary], feed_dict={image_placeholder: image})
      
      loss_grad_value /= loss_grad_value.std()+1e-8
      image += loss_grad_value*FLAGS.learning_rate
      
      # Update the events file.
      summary_str = sess.run(summary, feed_dict={image_placeholder: image})
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

      duration = time.time() - start_time
      print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))

      # Save an output image at the checkpoint intervls
      if (step + 1) % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
        save_image('%s-%s-%d-%04d.png' % (FLAGS.base_name, FLAGS.layer, FLAGS.channel, step), image[0])

if __name__ == '__main__':
  tf.app.run()

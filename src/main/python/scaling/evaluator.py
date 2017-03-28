
import tensorflow as tf
import numpy as np
import scipy.misc

import scaler


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('filter_sizes', '7,5,5', 'List: filter sizes')
flags.DEFINE_string('channels', '32,64', 'List: number of channels per hidden layers. '
                                         'Array size must be one less than filter_sizes')

flags.DEFINE_string('checkpoint_data','', 'Checkpoint data')

flags.DEFINE_string('source_image', '', 'Image to scale.')
flags.DEFINE_string('output_image', 'out.png', 'Where to save the output.')
flags.DEFINE_float('scale_amount', 2, 'Scaling factor')
flags.DEFINE_boolean('pre_scale', False, 'Whether to scale down the input before upscaling (for testing)')


def main(_):
  filter_sizes = list(map(int, FLAGS.filter_sizes.split(',')))
  channels = list(map(int, FLAGS.channels.split(',')))

  checkpoint_data = FLAGS.checkpoint_data
  
  source_image = scipy.misc.imread(FLAGS.source_image, mode='RGB').astype(np.float32)
  
  scale_amount = FLAGS.scale_amount
  full_resolution = None
  if FLAGS.pre_scale:
    full_resolution = (source_image.shape[0],source_image.shape[1])
    downscale_resolution = (int(source_image.shape[0]/scale_amount),int(source_image.shape[1]/scale_amount))
    source_image = scipy.misc.imresize(source_image, downscale_resolution)
    source_image = scipy.misc.imresize(source_image, full_resolution)
  else:
    full_resolution = (int(source_image.shape[0]*scale_amount),int(source_image.shape[1]*scale_amount))
    source_image = scipy.misc.imresize(source_image, full_resolution)
  
  with tf.Graph().as_default():
    image_scaler = scaler.Scaler(filter_sizes, channels)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_data)
    
    input_placeholder = tf.placeholder(tf.float32, shape=tuple([1,full_resolution[0],full_resolution[1],3]), name="input")
    output_placeholder = image_scaler.upscale(input_placeholder)
    
    output, = sess.run([output_placeholder], feed_dict={input_placeholder: np.array([source_image])})
    output_image = output[0]
    scipy.misc.imsave(FLAGS.output_image, output_image)


if __name__ == '__main__':
  tf.app.run()

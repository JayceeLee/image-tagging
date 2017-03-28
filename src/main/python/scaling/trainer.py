
#from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import time
import scipy.misc
import os
import random

import scaler


flags = tf.app.flags
FLAGS = flags.FLAGS

# Hyperparameters for the network
flags.DEFINE_string('filter_sizes', '7,5,5', 'List: filter sizes')
flags.DEFINE_string('channels', '32,64', 'List: number of channels per hidden layers. '
                                         'Array size must be one less than filter_sizes')

# Training details
flags.DEFINE_float('learning_rate', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('summary_interval', 1, 'How often to print summaries')
flags.DEFINE_integer('checkpoint_interval', 5, 'How often to save checkpoints')

# Training data
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('train_size', 300, 'Max size of train set.')
flags.DEFINE_integer('max_resolution', 512, 'Maximum resolution along one axis for the batch')
flags.DEFINE_string('train_dir', 'output/', 'Directory to put the training data.')
flags.DEFINE_string('image_dir', '/Users/ashmore/Downloads/101_ObjectCategories/', 'Where to find the images to train on.')
flags.DEFINE_string('checkpoint_data','', 'Checkpoint data')

# Model properties
flags.DEFINE_float('min_scale', 1.5, 'Minimum amount of upscaling to train')
flags.DEFINE_float('max_scale', 4.0, 'Maximum amount of upscaling to train')


def is_image(filename):
  filename = filename.lower()
  return filename.endswith('.png') or filename.endswith('.jpg')


def list_files(path):
  files = []
  for (dirpath, _, filenames) in os.walk(path):
    for filename in filenames:
      if is_image(filename):
        files.append(dirpath + '/' + filename)
  random.shuffle(files)
  return files


def load_image(path):
  return scipy.misc.imread(path, mode='RGB').astype(np.float32)


def get_images(path, batch_size):
  """Loads batch_size images randomly from path
  Returns: an array of np arrays of shape [width, height, 3]
           width and height may be different for each 
  """
  
  image_paths = list_files(path)[:batch_size]
  return [load_image(path) for path in image_paths]


def load_input(path, train_size, max_resolution):
  """
  Loads a batch of high-resolution images. Images are scaled down to have a common resolution.
  Output: tensor of shape [batch_size, image_width, image_height, 3]
  """
  
  # An array of images that each can have a different size
  images = get_images(path, train_size)
  
  # Use aspect ratio of the 1st image (totally arbitrary)
  # Assume shape[0] is width shape[1] is height
  aspect_ratio = images[0].shape[1] / images[0].shape[0]
  
  if aspect_ratio < 1:
    # width is wider
    width = min(images[0].shape[0], max_resolution)
    height = int(aspect_ratio * width)
  else:
    height = min(images[0].shape[1], max_resolution)
    width = int(height / aspect_ratio)
  
  # resize all images to width,height
  # This will scale the images outside their preferred aspect ratio. Ideally I'd like to crop
  # somehow, but the way to do this is not quite clear.
  return np.array([scipy.misc.imresize(image, (width, height)) for image in images])


def apply_distortion(image, min_scale, max_scale):
  """
  Apply some kind of distortion/downsampling to the image
  Input:
    image: np array of shape [width, height, 3]
  Output: distorted image of same shape as input.
  """
  width, height = image.shape[0], image.shape[1]
  scale_factor = random.uniform(min_scale, max_scale)
  downscaled_image = scipy.misc.imresize(image, (int(width/scale_factor), int(height/scale_factor)))
  # could apply some noise/artifacts at this point
  upscaled_image = scipy.misc.imresize(downscaled_image, (width, height))
  return upscaled_image


def train_batch(scaler, input_images, distorted_images, learning_rate):
  upscaled_images = scaler.upscale(distorted_images)
  loss = tf.nn.l2_loss(input_images - upscaled_images)
  return loss, tf.train.AdamOptimizer(learning_rate).minimize(loss)


def main(_):
  
  filter_sizes = list(map(int, FLAGS.filter_sizes.split(',')))
  channels = list(map(int, FLAGS.channels.split(',')))
  learning_rate = FLAGS.learning_rate
  max_steps = FLAGS.max_steps
  summary_interval = FLAGS.max_steps
  checkpoint_interval = FLAGS.checkpoint_interval
  batch_size = FLAGS.batch_size
  train_size = FLAGS.train_size
  max_resolution = FLAGS.max_resolution
  train_dir = FLAGS.train_dir
  image_dir = FLAGS.image_dir
  checkpoint_data = FLAGS.checkpoint_data
  min_scale = FLAGS.min_scale
  max_scale = FLAGS.max_scale
  
  with tf.Graph().as_default():
  
    image_scaler = scaler.Scaler(filter_sizes, channels)
    input_images = load_input(image_dir, train_size, max_resolution)
    distorted_images = np.array([apply_distortion(image, min_scale, max_scale) for image in input_images])
    
    print('Input image shape: %s' % (input_images,))

    input_placeholder = tf.placeholder(tf.float32, shape=tuple([batch_size,input_images.shape[1],input_images.shape[2],3]), name="input")
    distorted_placeholder = tf.placeholder(tf.float32, shape=tuple([batch_size,input_images.shape[1],input_images.shape[2],3]), name="distorted")
    loss, train_op = train_batch(image_scaler, input_placeholder, distorted_placeholder, learning_rate)

    sess = tf.Session()
    
    saver = tf.train.Saver()
    
    tf.summary.scalar(loss.op.name, loss)
    image_scaler.summarize_tensors()
    summary_merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    if checkpoint_data:
      saver.restore(sess, train_dir + checkpoint_data)
    else:
      sess.run(tf.global_variables_initializer())

    partitioned_input_images = [input_images[i:i + batch_size,:,:,:] for i in range(0, input_images.shape[0], batch_size)]
    partitioned_distorted_images = [distorted_images[i:i + batch_size,:,:,:] for i in range(0, distorted_images.shape[0], batch_size)]

    for step in range(max_steps):
      start_time = time.time()
      image_batch = partitioned_input_images[step % len(partitioned_input_images)]
      distorted_batch = partitioned_distorted_images[step % len(partitioned_input_images)]
      feed_dict = {input_placeholder: image_batch, distorted_placeholder: distorted_batch}

      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

      if step % summary_interval == 0:
        summary, = sess.run([summary_merged], feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
        summary_writer.flush()

      duration = time.time() - start_time
      print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))
      
      if (step + 1) % checkpoint_interval == 0 or (step + 1) == max_steps:
          print('Saving checkpoint.')
          checkpoint_file = os.path.join(train_dir, 'checkpoint')
          saver.save(sess, checkpoint_file, global_step=step)

if __name__ == '__main__':
  tf.app.run()

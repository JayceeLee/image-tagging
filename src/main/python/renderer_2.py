
""" Attempt to construct an image based on an input classification """

import tensorflow as tf
import numpy as np
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy.misc

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('layers', 'conv4_2', 'List: Layer to use')
flags.DEFINE_string('channels', '123', 'List: Channel to for the layer')
flags.DEFINE_string('scales', '1', 'List: Scale for convolution')
flags.DEFINE_string('weights', '1', 'List: Weight of the convolution')

flags.DEFINE_float('learning_rate', .1, 'Learning rate')
flags.DEFINE_string('output_dir', './output', 'Directory to output stuff.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('checkpoint_interval', 1, 'How often to save output')
flags.DEFINE_string('base_name', 'image', 'Base name of the image')
flags.DEFINE_string('base_image', '', 'Image to use as basis')
flags.DEFINE_integer('image_height', 512, 'Height of image if using noise')
flags.DEFINE_integer('image_width', 512, 'Width of image if using noise')
flags.DEFINE_bool('use_laplacian_norm', True, 'Whether to use the Laplacian norm')
flags.DEFINE_integer('patch_size', 512, 'Size of the image patch to use')
flags.DEFINE_integer('patch_edge_size', 16, 'interpolation edge of the patch')


import vgg19_conv as vgg
import laplacian


def save_image(filename, data):
  scipy.misc.imsave('%s/%s' % (FLAGS.output_dir, filename), data)


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
@tffunc(np.float32, np.int32)
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]


def vgg_at_resolution(image, patch_size, vgg_scale, weight, layer, channel):
  """ 
  Args:
    image: tensor of shape [batch, patch_size, patch_size, channels]
    vgg_scale: a float that is < 1
  """
  
  original_size = [patch_size, patch_size]
  scaled_size = [int(original_size[0] * vgg_scale), int(original_size[1] * vgg_scale)]
  scaled_image = tf.image.resize_bilinear(image, scaled_size)
  
  with tf.name_scope('vgg'):
      net = vgg.Vgg19()
      net.build(scaled_image)

  score = weight * tf.reduce_mean(getattr(net, layer)[:,:,:,channel])
  score_grad = tf.gradients(score, scaled_image)[0]

  scaled_grad = tf.image.resize_bilinear(score_grad, original_size)
  return score, scaled_grad


def patch_filter_curve(i, patch_size, patch_edge_size, min, max):
  if i < patch_edge_size and not min:
    return 1 - (patch_edge_size - i)/patch_edge_size
  if i > patch_size - patch_edge_size and not max:
    return 1 - (i - patch_size + patch_edge_size + 1)/patch_edge_size
  return 1.0


def build_patch_filter(patch_size, patch_edge_size, x_min, x_max, y_min, y_max):
  filter = np.ones((1, patch_size, patch_size, 3))
  for i in range(0, patch_size):
    for j in range(0, patch_size):
      x_factor = patch_filter_curve(i, patch_size, patch_edge_size, x_min, x_max)
      y_factor = patch_filter_curve(j, patch_size, patch_edge_size, y_min, y_max)
      filter[:,i,j,:] = min(x_factor, y_factor)
  return filter


def clamp(value, min_val, max_val):
  return np.maximum(np.minimum(value, max_val), min_val)


def main(_):
  layers = FLAGS.layers.split(',')
  channels = list(map(int, FLAGS.channels.split(',')))
  scales = list(map(float, FLAGS.scales.split(',')))
  weights = list(map(float, FLAGS.weights.split(',')))
  patch_size = FLAGS.patch_size
  patch_edge_size = FLAGS.patch_edge_size
  
  assert(len(layers) == len(channels))
  assert(len(layers) == len(scales))
  assert(len(layers) == len(weights))
  
  with tf.Graph().as_default():
    if FLAGS.base_image:
      base_image = scipy.misc.imread(path, mode='RGB').astype(np.float32)
      image = np.array([base_image], dtype = np.float32)
    else:
      image = np.random.uniform(size=(1,FLAGS.image_width,FLAGS.image_height,3)).astype(np.float32) + 100.0

    patch_position = tf.placeholder(tf.int32, shape=(2))
    patch_x = patch_position[0]
    patch_y = patch_position[1]

    image_placeholder = tf.placeholder(tf.float32, shape=image.shape, name='image_pl')
    image_patch = image_placeholder[:,patch_x:patch_x+patch_size,patch_y:patch_y+patch_size,:]
    
    score = 0
    score_grad = tf.zeros_like(image_patch)
    for i in range(0, len(layers)):
      layer_score, layer_score_grad = vgg_at_resolution(
          image_patch, patch_size, scales[i], weights[i], layers[i], channels[i])
      score = score + layer_score
      score_grad = score_grad + layer_score_grad
      
    if FLAGS.use_laplacian_norm:
      score_grad = laplacian.lap_normalize(score_grad)
    else:
      score_grad /= score_grad_value.std()+1e-8

    sess = tf.Session()
    
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      
      # IMAGE DIMENSIONS ARE REVERSED!?!?!
      # WHO PUTS HEIGHT AS FIRST DIMENSION????????
      
      # Also: Why does image get gradually lighter over time with no clear operation that would affect that?
      # Does not appear to be an issue with the actual numbers. Maybe something is happening when the image is saved
      
      width_minus_patch = FLAGS.image_width-patch_size
      height_minus_patch = FLAGS.image_height-patch_size
      px = np.random.randint(-patch_edge_size,width_minus_patch+patch_edge_size)
      py = np.random.randint(-patch_edge_size,height_minus_patch+patch_edge_size)
      px = clamp(px, 0, width_minus_patch-1)
      py = clamp(py, 0, height_minus_patch-1)
      
      x_min = px == 0
      x_max = px == width_minus_patch-1
      y_min = py == 0
      y_max = py == height_minus_patch-1
      patch_position_value = np.array([px, py])

      patch_grad_value, score_value = sess.run([score_grad, score],
                                          feed_dict={image_placeholder: image,
                                                     patch_position: patch_position_value})

      patch_filter = build_patch_filter(patch_size, FLAGS.patch_edge_size, x_min, x_max, y_min, y_max)
      #image += score_grad_value * FLAGS.learning_rate
      image[:,px:px+patch_size,py:py+patch_size,:] += patch_grad_value * patch_filter * FLAGS.learning_rate
      
#      image = clamp(image, 0, 255)

      duration = time.time() - start_time
      print('Step %d: score = %.3f (%.3f sec)' % (step, score_value, duration))
      print('%d %d, %.1f' % (px, py, image[0,0,0,0]))

      # Save an output image at the checkpoint intervls
      if (step + 1) % FLAGS.checkpoint_interval == 0 or (step + 1) == FLAGS.max_steps:
        save_image('%s-%04d.png' % (FLAGS.base_name, step), image[0])


if __name__ == '__main__':
  tf.app.run()

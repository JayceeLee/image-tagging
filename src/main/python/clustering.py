
"""
Attempt to compare input images and cluster them using vgg19
"""

import tensorflow as tf
import numpy as np
import os
import random

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('cluster_dir', '../../../output/', 'Directory where the images to cluster live.')
flags.DEFINE_string('output_file', 'out.npy', 'Output file.')
flags.DEFINE_integer('batch_size', 20, 'Batch size.')
flags.DEFINE_integer('save_interval', 1, 'Save interval.')

import dataset
import vgg19_trainable as vgg

# Steps:
# 1) load data
# 2) perform analysis on data in batches
# 3) save data somewhere; path + vgg values
#    This lets us do transformation not all at once
# 4 - in clustering_tsne) perform TSNE to visualize clusters


def is_image(filename):
  filename = filename.lower()
  return filename.endswith('.png') or filename.endswith('.jpg')


def list_files(path):
  files = []
  for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
      if is_image(filename):
        files.append(dirpath + '/' + filename)
  random.shuffle(files)
  return files


def chunks(l, n):
  # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]


def load_images(filenames):
  return np.array([dataset.imread(filename) for filename in filenames])


def save_data(data):
  np.save(FLAGS.output_file, data)


def run_vgg():
  all_files = list_files(FLAGS.cluster_dir)
  batched_filenames = chunks(all_files, FLAGS.batch_size)
  all_output = {}
  total_progress = 0
  
  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=tuple([FLAGS.batch_size, 224, 224, 3]), name="images")

    with tf.name_scope('vgg'):
      # TODO: set trainable to false. There's something weird in the lib that prevents data from getting correctly loaded as a constant.
#      net = vgg.Vgg19(trainable = False)
      net = vgg.Vgg19(trainable = True)
      net.build(images_placeholder, train_mode=tf.constant(False))
    result = net.relu7
  
    with tf.Session() as sess:
      
      sess.run(tf.initialize_all_variables())
      
      for batch in batched_filenames:
        images = load_images(batch)
        batch_data, = sess.run([result], feed_dict = {images_placeholder: images})
        
        for i in range(0, FLAGS.batch_size):
          filename = batch[i]
          filename_data = batch_data[i]
          all_output[filename] = filename_data
          
        print('completed batch %d' % (total_progress))
        total_progress = total_progress + 1
        if total_progress % FLAGS.save_interval == 0:
          save_data(all_output)

    save_data(all_output)


def main(_):
  run_vgg()


if __name__ == '__main__':
  tf.app.run()

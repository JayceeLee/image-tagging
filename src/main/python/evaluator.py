
import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('weights1', 100, 'Number of neurons in first fully connected layer.')
flags.DEFINE_integer('weights2', 100, 'Number of neurons in second fully connected layer.')
flags.DEFINE_string('train_dir', '../../../output/', 'Directory to put the training data.')
flags.DEFINE_string('image_path','../../../output/out000.png', 'Image to evaluate')
flags.DEFINE_string('checkpoint_data','checkpoint-4', 'Checkpoint data')

import dataset
import classifier

IMAGE_SHAPE = (1,224,224,3)


def main(_):
  input = np.array([dataset.imread(FLAGS.image_path)], dtype = np.float32)
  indices_to_tags,_ = dataset.load_tag_data(FLAGS.train_dir)

  with tf.Graph().as_default():
    images_placeholder = tf.placeholder(tf.float32, shape=IMAGE_SHAPE)
    logits = classifier.inference(
        images_placeholder, len(indices_to_tags), FLAGS.weights1, FLAGS.weights2)
    
    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.train_dir + FLAGS.checkpoint_data)
      (raw_result,) = sess.run([logits], {images_placeholder: input})
      result = {tag: raw_result[0,index] for index,tag in indices_to_tags.items()}
      print(result)


if __name__ == "__main__":
  tf.app.run()

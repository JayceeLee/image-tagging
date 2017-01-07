
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

import dataset

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_file', 'out.npy', 'Output file.')


def load_data():
  """Load data and break into an array of labels, and a list of data values
  labels has shape (# data points)
  data has shape (# data points, 1000)
  """
  loaded_data = np.load(FLAGS.data_file)[()]
  
  labels = []
  data = []
  
  for filename in loaded_data:
    labels.append(filename)
    data.append(loaded_data[filename])
  return np.array(labels), np.array(data)


def plot_with_labels(lowDWeights, labels, filename='tsne.png'):
  assert lowDWeights.shape[0] >= len(labels), "More labels than weights"
  plt.figure(dpi=300)  # inches at 100dpi
  
  fig, ax = plt.subplots()
  
  for i, label in enumerate(labels):
    x, y = lowDWeights[i,:]
    plt.scatter(x, y)

    image = dataset.imread(label, (32,32))
    offset_image = OffsetImage(image, zoom=1)
    ax.add_artist(AnnotationBbox(offset_image, (x, y), xycoords='data', frameon=False))

  plt.savefig(filename, dpi=300)


def plot_clusters():
  labels, data = load_data()
  print('Data size: %d' % (len(labels)))
  print('Running t-SNE...')
  tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
  weights = tsne.fit_transform(data)
  print('Creating plot...')
  plot_with_labels(weights, labels)


def main(_):
  plot_clusters()


if __name__ == '__main__':
  tf.app.run()


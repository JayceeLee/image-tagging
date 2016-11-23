
import scipy.misc
import numpy as np

IMAGE_SIZE = 224

def imread(path, target_size=(IMAGE_SIZE,IMAGE_SIZE)):
  return scipy.misc.imresize(scipy.misc.imread(path).astype(np.float32), target_size)

def load_raw_data(path):
  all_images = list()
  all_tags = list()
  
  tags_to_indices = dict()
  indices_to_tags = dict()
  tag_index = 0
  
  with open(path + 'tags.txt', 'r') as tags_file:
    for line in tags_file:
      # Maybe have something to limit # of lines read?
      bits = line.replace('[','').replace(']','').replace(',','').split()
      file_name = bits[0]
      tags = bits[1:]
      all_images.append(imread(path + file_name))
      all_tags.append(tags)
      for tag in tags:
        if tag not in tags_to_indices:
          tags_to_indices[tag] = tag_index
          indices_to_tags[tag_index] = tag
          tag_index = tag_index+1
  
  tag_labels = list()
  for tags in all_tags:
    row = np.zeros(len(indices_to_tags), dtype = np.float32)
    for tag in tags:
      row[tags_to_indices[tag]] = 1
    tag_labels.append(row)

  return indices_to_tags, np.array(all_images, dtype = np.float32), np.array(tag_labels)

def load_data(path = None, validation_fraction = .25):
  """ load data 
  Args:
    validation_fraction: fraction of data to use for validation & testing
    test_fraction: fraction of validation_data to use as the "test" category
  Returns:
    train_data, validation_data, test_data
  """
  
  if not path:
    path = '../../../output/'
  
  indices_to_tags, images, labels = load_raw_data(path)
  
  count = len(images)
  validation_count = int(count * validation_fraction)
  train_count = count - validation_count
  
  return (DataSet(indices_to_tags, images[:train_count], labels[:train_count]),
          DataSet(indices_to_tags, images[train_count:], labels[train_count:]))

class DataSet(object):
  def __init__(self, indices_to_tags, images, labels):
    self._images = images
    self._labels = labels
    self._indices_to_tags = indices_to_tags
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = images.shape[0]
  
  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels
  
  @property
  def indices_to_tags(self):
    return self._indices_to_tags
  
  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


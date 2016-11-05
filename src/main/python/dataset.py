
import scipy.misc
import numpy as np

def imread(path, target_size=(200,200)):
  return scipy.misc.imresize(
    scipy.misc.imread(path).astype(np.float),
    target_size)

def load_data(path):
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
    row = np.zeros(len(indices_to_tags), dtype = np.int32)
    for tag in tags:
      row[tags_to_indices[tag]] = 255
    tag_labels.append(row)

  return (indices_to_tags, np.array(all_images), np.array(tag_labels))

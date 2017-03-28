
import tensorflow as tf


class Scaler:
  
  def __init__(self, filter_sizes, channels, train_mode=False):
    """
    Params:
      filter_sizes: list of type int with size that is the # of layers
      channels: list of type int with the # of channels per hidden layer (1 layer less than filter_sizes)
      train_mode: whether we are training the network. If true, use nn dropout
    """
    
    channels.append(3)
    assert(len(channels) == len(filter_sizes))
    
    self.filters = list()
    self.biases = list()
    self.train_mode = train_mode
    
    for layer_index in range(len(filter_sizes)):
      with tf.variable_scope("var_layer_"+str(layer_index)):
        filter_size = filter_sizes[layer_index]
        in_channels = channels[layer_index-1]
        out_channels = channels[layer_index]
        
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = tf.Variable(initial_value, name="filters_"+str(layer_index))
        self.filters.append(filters)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = tf.Variable(initial_value, name="biases_"+str(layer_index))
        self.biases.append(biases)

  def summarize_tensors(self):
    for layer_index in range(len(self.filters)):
      tf.summary.histogram(self.filters[layer_index].name, self.filters[layer_index])
      tf.summary.histogram(self.biases[layer_index].name, self.biases[layer_index])
  
  def conv_layer(self, input_layer, layer_index, name):
    with tf.variable_scope(name):
      conv = tf.nn.conv2d(input_layer, self.filters[layer_index], [1, 1, 1, 1], padding='SAME')
      bias = tf.nn.bias_add(conv, self.biases[layer_index])
      relu = tf.nn.relu(bias)
      if (self.train_mode):
        relu = tf.nn.dropout(relu, 0.5)
      return relu

  def upscale(self, input_image):
    """
    Params:
      input_image: tensor with shape [batch, x, y, 3]. Input is assumed to have been upsampled already
    Output:
      tensor with the same shape as input_image
    """
    
    current_layer = input_image
    
    for layer_index in range(len(self.filters)):
      current_layer = self.conv_layer(current_layer,
                                      layer_index,
                                      "layer_"+str(layer_index))
    return current_layer


from subprocess import call

base_image = None

channels_per_layer = {
  'conv2_1': 128,
  'conv2_2': 128,
  'conv3_1': 256,
  'conv3_2': 256,
  'conv3_3': 256,
  'conv3_4': 256,
  'conv4_1': 512,
  'conv4_2': 512,
  'conv4_3': 512,
  'conv4_4': 512,
  'conv5_1': 512,
  'conv5_2': 512,
  'conv5_3': 512,
  'conv5_4': 512,
#  'fc6': 4096,
#  'fc7': 4096,
#  'fc8': 1000,
}

def execute(layer, channel):
  command = [
    'python3',
    'renderer_1.py',
    '--channel=%d' % (channel),
    '--layer=%s' % (layer),
    '--learning_rate=.5',
    '--checkpoint_interval=1',
    '--output_dir=./output/bulk'
  ]
  
  if base_image:
    command.append('--base_image=' + base_image)
  
  print('>>> ' + ' '.join(command))
  call(command)


if __name__ == "__main__":  
  for layer, count in channels_per_layer.items():
    for channel in range(count):
      execute(layer, channel)


from subprocess import call

base_input_name = 'squirrel'
base_output_name = 'squirrel'

channels_per_layer = {
#  'conv2_1': 128,
#  'conv2_2': 128,
#  'conv3_1': 256,
#  'conv3_2': 256,
#  'conv3_3': 256,
#  'conv3_4': 256,
  'conv4_1': 512,
#  'conv4_2': 512,
#  'conv4_3': 512,
#  'conv4_4': 512,
#  'conv5_1': 512,
#  'conv5_2': 512,
#  'conv5_3': 512,
#  'conv5_4': 512,
#  'fc6': 4096,
#  'fc7': 4096,
#  'fc8': 1000,
}

def save_video(layer, channel):
  command = [
    'avconv',
    '-i',
    'output/bulk/%s-%s-%d-%%04d.png' % (base_input_name, layer, channel),
    '-r',
    '30',
    'output/video/%s-%s-%d.mp4' % (base_output_name, layer, channel),
  ]
  print('>>> ' + ' '.join(command))
  call(command)


if __name__ == "__main__":
  for layer in sorted(channels_per_layer.keys()):
    for channel in range(channels_per_layer[layer]):
    	save_video(layer, channel)


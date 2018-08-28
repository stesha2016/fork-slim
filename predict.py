# created by stesha
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import json
import numpy as np

from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets import dataset_utils
from datasets import mydata

from preprocessing import inception_preprocessing

slim = tf.contrib.slim

_NUM_CLASSES = 243
INFO_FILE = 'coin_info.json'

merge_info = {'kmmerge123_back': ['km1_back', 'km2_back', 'km3_back'],
              'kmmerge341342343_back': ['km341_back', 'km342_back', 'km343_back'],
              'kmmerge881882_back': ['km881_back', 'km882_back'],
              'kmmerge980981_back': ['km980_back', 'km981_back'],
              'kmmerge982983_back': ['km982_back', 'km983_back'],
              'kmmerge14631464_back': ['km1463_back', 'km1464_back'],
              'kmmerge15241525_back': ['km1524_back', 'km1525_back'],
              'kmmerge15261527_back': ['km1526_back', 'km1527_back'],
              'kmmerge15761578_back': ['km1576_back', 'km1578_back'],
              'newmerge345_back': ['new3_back', 'new4_back', 'new5_back']}

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/data/train_logs/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '/data/mydata/',
    'The directory where the dataset label file and info file are stored.')

tf.app.flags.DEFINE_string(
    'predict_file', None, 'The file to be predict.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS

def read_info_file(dataset_dir, filename=INFO_FILE):
  infos_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(infos_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  class_names_to_info = {}
  for line in lines:
    dict_info = json.loads(line)
    coin_info = {}
    coin_info['info'] = dict_info['info']
    coin_info['value'] = dict_info['value']
    class_names_to_info[dict_info['name']] = coin_info
  return class_names_to_info


def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels

def main(_):
  if not FLAGS.predict_file:
    raise ValueError('You must supply the predict file path with --predict_file')

  with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

##############################################################################################################
    batch_size = 1
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=_NUM_CLASSES,
        is_training=False)
    image_size = network_fn.default_image_size

    # images, images_raw = load_batch_2(FLAGS.predict_file, height=image_size, width=image_size)
    image_string = tf.gfile.FastGFile(FLAGS.predict_file, 'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    logits, _ = network_fn(processed_images)
    probabilities = tf.nn.softmax(logits)

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_variables_to_restore())

    with tf.Session() as sess:
      with slim.queues.QueueRunners(sess):
        sess.run(tf.initialize_local_variables())
        init_fn(sess)
        np_images, np_probabilities = sess.run([image, probabilities])
        predicted_label = np.argmax(np_probabilities[0, :])
        print(predicted_label)

    labels_to_names = None
    if dataset_utils.has_labels(FLAGS.dataset_dir):
      labels_to_names = dataset_utils.read_label_file(FLAGS.dataset_dir)

    class_name = labels_to_names[predicted_label]
    print(class_name)

    if dataset_utils.has_labels(FLAGS.dataset_dir, INFO_FILE):
      class_names_to_info = read_info_file(FLAGS.dataset_dir)

    if class_name in merge_info:
      print('It is the back of a coin, please take a photo of front.')
      class_name_list = merge_info[class_name]
      for item in class_name_list:
        info = class_names_to_info[item.split('_')[0]]
        print('Maybe the value is {}'.format(info['value']))
    else:
      name_info = class_name.split('_')
      origin_name = name_info[0]
      back_or_front = name_info[1]    

      info = class_names_to_info[origin_name]
      print('value is %s, it is %s of the coin, the other info is %s' % (info['value'], back_or_front, info['info']))


  ########################################################################################################################

    # network_fn = nets_factory.get_network_fn(
    #   FLAGS.model_name,
    #   num_classes=_NUM_CLASSES,
    #   is_training=True)
    # image_size = network_fn.default_image_size
    # batch_size = 1

    # with tf.Graph().as_default():
    #   tf.logging.set_verbosity(tf.logging.INFO)
    
    #   dataset = mydata.get_split('validation', FLAGS.dataset_dir)
    #   images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)
    #   logits, _ = network_fn(images)

    #   probabilities = tf.nn.softmax(logits)
    
    #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    #   init_fn = slim.assign_from_checkpoint_fn(
    #     checkpoint_path,
    #     slim.get_variables_to_restore())
    
    #   with tf.Session() as sess:
    #     with slim.queues.QueueRunners(sess):
    #       sess.run(tf.initialize_local_variables())
    #       init_fn(sess)
    #       np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])
    
    #       for i in range(batch_size): 
    #         image = np_images_raw[i, :, :, :]
    #         true_label = np_labels[i]
    #         predicted_label = np.argmax(np_probabilities[i, :])
    #         print('true is {}, predict is {}'.format(true_label, predicted_label))


if __name__ == '__main__':
  tf.app.run()

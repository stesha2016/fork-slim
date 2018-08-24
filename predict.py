# created by stesha
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import cv2

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

_NUM_CLASSES = 258

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'predict_file', None, 'The file to be predict.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.predict_file:
    raise ValueError('You must supply the predict file path with --predict_file')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(_NUM_CLASSES - FLAGS.labels_offset),
        is_training=False)

    preprocessing_name = FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    image_size = network_fn.default_image_size
    image = cv2.imread(FLAGS.predict_file)
    image = cv2.resize(image, (image_size, image_size))
    need_predict_list = []
    need_predict_list.append(image)

    images = tf.placeholder(dtype=tf.float32, shape=[None, image_size, image_size, 3])
    logits, _ = network_fn(images)
    predictions = tf.argmax(logits, 1)
    print(predictions)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, checkpoint_path)

    _predictions = sess.run(predictions, feed_dict={images:need_predict_list})
    print(_predictions)


if __name__ == '__main__':
  tf.app.run()

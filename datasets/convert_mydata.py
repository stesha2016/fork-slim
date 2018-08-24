# Created by stesha
import tensorflow as tf
import os
import random
import math
import sys

_NUM_SHARDS = 258
_RANDOM_SEED = 0
_NUM_VALIDATION = 258

from datasets import dataset_utils

class ImageReader(object):
  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    # convert image data to uint8 tenser
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
    assert(len(image.shape) == 3)
    assert(image.shape[2] == 3)
    return image

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'mydata_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def _get_filenames_and_classes(dataset_dir):
  mydata_root = dataset_dir
  directories = []
  class_names = []
  for filename in os.listdir(mydata_root):
    path = os.path.join(mydata_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)

def _clean_up_temporary_files(dataset_dir):
  for file in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, file)
    if os.path.isdir(path):
      tf.gfile.DeleteRecursively(path)

def _convert_dataset(split_name, filename, class_names_to_ids, dataset_dir):
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filename) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()
    with tf.Session('') as sess:
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(filename))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filename), shard_id))
            sys.stdout.flush()

            image_data = tf.gfile.FastGFile(filename[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filename[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

def run(dataset_dir):
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir)

  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  _clean_up_temporary_files(dataset_dir)
  print('\n Finished converting the mydata dataset')

# created by stesha
import argparse

import numpy as np
import tensorflow as tf

from preprocessing import inception_preprocessing


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224):
  image_string = tf.gfile.FastGFile(file_name, 'rb').read()
  image = tf.image.decode_jpeg(image_string, channels=3)
  processed_image = inception_preprocessing.preprocess_image(image, input_height, input_width, is_training=False)
  processed_images  = tf.expand_dims(processed_image, 0)  

  sess = tf.Session()
  result = sess.run(processed_images)
  print(result.shape)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label


if __name__ == "__main__":
  file_name = "./backup/mydata/km335_back/km335_back.jpg"
  model_file = \
    "./data/train_logs_1/freeze_graph_5002.pb"
  label_file = "./data/mydata/labels.txt"
  input_height = 224
  input_width = 224
  input_layer = "MobilenetV2/input"
  output_layer = 'MobilenetV2/Predictions/Reshape_1'

  graph = load_graph(model_file)
  # with open('./data/train_logs_1/operations_5000.txt', 'wb') as f:
  #   for op in graph.get_operations():
  #     f.writelines(str(op.name) + ',' + str(op.values()) + '\n')

  t = read_tensor_from_image_file(
      file_name,
      input_height=input_height,
      input_width=input_width)

  input_name = "import/" + input_layer
  output_name = "import/" + output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  with tf.Session(graph=graph) as sess:
    results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
    })
  results = np.squeeze(results)
  print(results.shape)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  for i in top_k:
    print(labels[i], results[i])

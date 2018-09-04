# import tensorflow as tf

# meta_path = './data/train_logs_1/model.ckpt-5001.meta' # Your .meta file

# with tf.Session() as sess:

#     # Restore the graph
#     saver = tf.train.import_meta_graph(meta_path)

#     # Load weights
#     saver.restore(sess, tf.train.latest_checkpoint('./data/train_logs_1'))

#     graph = tf.get_default_graph()

#     with open('./data/train_logs_1/operations_5001.txt', 'wb') as f:
#         for op in graph.get_operations():
#             f.writelines(str(op.name) + ',' + str(op.values()) + '\n')



import tensorflow as tf

meta_path = './data/train_logs_1/model.ckpt-5002.meta' # Your .meta file
output_node_names = ['MobilenetV2/Predictions/Reshape_1']    # Output nodes

with tf.Session() as sess:

    # Restore the graph
    saver = tf.train.import_meta_graph(meta_path)

    # Load weights
    saver.restore(sess, tf.train.latest_checkpoint('./data/train_logs_1'))

    # Freeze the graph
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        output_node_names)

    # Save the frozen graph
    with open('./data/train_logs_1/freeze_graph_5002.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())
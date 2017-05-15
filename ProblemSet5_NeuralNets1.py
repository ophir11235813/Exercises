import tensorflow as tf
import numpy as np

graph = tf.Graph()

with graph.as_default():
    _x = tf.random_normal([8,10]) # Make a random matrix
    x  = tf.convert_to_tensor(_x)
    # weights = tf.truncated_normal([10,2],0,0.1, dtype = np.float32)
    # y  = tf.matmul(x,weights)
    y = tf.contrib.layers.fully_connected(inputs = x,
                                          num_outputs = 3,
                                          activation_fn = tf.nn.sigmoid,
                                          weights_initializer = tf.contrib.layers.xavier_initializer())


with tf.Session(graph = graph) as sess:
    sess.run(tf.global_variables_initializer())
    print(y.eval())


tf.reset_default_graph()

with graph.as_default():
    x = tf.random_uniform(shape = (2,3,3,3), dtype = np.float32)
    filter = tf.get_variable("filter", shape=(2,2), )

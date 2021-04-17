import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    tf.random.set_seed(123)
    src = tf.random.normal(shape=[3, 3])
    mapping = tf.constant([[2], [1], [0]])
    print(mapping.shape)
    print(src)
    src = tf.tensor_scatter_nd_update(src, mapping, src)
    print(src)
    # mapping = tf.constant([4, 0, 7, 5, 8, 3, 1, 6, 9, 2], dtype=tf.int32)
    # mapping = tf.reshape(mapping, shape=[10, 1])
    # print(mapping)
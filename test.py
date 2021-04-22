import tensorflow as tf
import numpy as np

if __name__ == '__main__':
        a = tf.random.uniform([10, 3])
        print(a)
        mapping = np.arange(10)
        np.random.shuffle(mapping)

        mapping = tf.convert_to_tensor(mapping, dtype=tf.int32)
        mapping = tf.reshape(mapping, shape=[10, 1])
        print(mapping.shape)
        print(mapping)
        a = tf.tensor_scatter_nd_update(a, mapping, a)
        print(a)
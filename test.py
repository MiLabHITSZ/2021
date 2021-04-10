import tensorflow as tf

if __name__ == '__main__':
    a = tf.constant([1], shape=[1, 1])
    print(tf.shape(a))
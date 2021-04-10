import tensorflow as tf

if __name__ == '__main__':
    a = tf.constant([[1]], shape=[1, 1])
    b = a.numpy()
    print(type(b))
    print(tf.shape(a))
    print(a[0][0])

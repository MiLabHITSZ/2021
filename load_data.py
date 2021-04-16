import tensorflow as tf
from tensorflow.keras import datasets
from attack import *


def preprocess_mnist(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def preprocess_cifar10(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def load_mnist():
    (x, y), (x_test_in, y_test_in) = datasets.mnist.load_data()

    # 合成恶意数据进行CAP攻击
    mal_x_out, mal_y_out = mal_data_synthesis(x_test_in, 2, 4)
    # 对合成的恶意数据进行拼接
    x = np.vstack((x, mal_x_out))
    y = np.append(y, mal_y_out)

    train_db_in = tf.data.Dataset.from_tensor_slices((x, y))
    train_db_in = train_db_in.shuffle(10000)
    train_db_in = train_db_in.batch(128)
    train_db_in = train_db_in.map(preprocess_mnist)

    x_test_in = tf.convert_to_tensor(x_test_in)
    x_test_in = tf.cast(x_test_in, dtype=tf.float32) / 255
    x_test_in = tf.reshape(x_test_in, [-1, 28 * 28])
    y_test_in = tf.convert_to_tensor(y_test_in)
    y_test_in = tf.cast(y_test_in, dtype=tf.int64)

    mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32)/255
    mal_x_out = tf.reshape(mal_x_out, [-1, 28*28])
    return train_db_in, x_test_in, y_test_in, mal_x_out


def load_cifar10():
    (x, y), (x_test_in, y_test_in) = datasets.cifar10.load_data()
    y = tf.squeeze(y, axis=1)
    y_test_in = tf.squeeze(y_test_in, axis=1)
    print(x.shape, y.shape, x_test_in.shape, y_test_in.shape)
    train_db_in = tf.data.Dataset.from_tensor_slices((x, y))
    train_db_in = train_db_in.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test_in, y_test_in))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    return train_db_in, test_db


if __name__ == '__main__':
    # load_cifar10()
    load_mnist()
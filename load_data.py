import tensorflow as tf
from tensorflow.keras import datasets
from attack import *
from defend import *
import numpy as np

def preprocess_mnist(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    x_in = tf.reshape(x_in, [-1, 28 * 28])
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def merge_mnist_fnn(x_train_in, y_train_in, x_test_in, y_test_in, x_mal_in, y_mal_in, epoch):
    x_train, y_train, x_test, y_test, x_mal, y_mal = x_train_in, y_train_in, x_test_in, y_test_in, x_mal_in, y_mal_in

    # 随机生成mapping
    np.random.seed(epoch)
    mapping = np.arange(10)
    np.random.shuffle(mapping)
    print(mapping)
    #
    y_train = defend_cap_attack(y_train, mapping)

    # 对合成的恶意数据进行拼接
    x_train = np.vstack((x_train, x_mal))
    y_train = np.append(y_train, y_mal)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    x_mal = tf.convert_to_tensor(x_mal, dtype=tf.float32) / 255
    x_mal = tf.reshape(x_mal, [-1, 28 * 28])

    return train_db, test_db, x_mal, mapping



def preprocess_cifar10(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def load_mnist_fnn():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 进行cap攻击防御
    # mapping = np.array([[4], [0], [7], [5], [8], [3], [1], [6], [9], [2]])
    # y_train = defend_cap_attack(y_train, mapping)
    # 合成恶意数据进行CAP攻击
    mal_x_out, mal_y_out = mal_mnist_fnn_synthesis(x_test, 2, 4)
    # 对合成的恶意数据进行拼接
    x_train = np.vstack((x_train, mal_x_out))
    y_train = np.append(y_train, mal_y_out)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32) / 255
    mal_x_out = tf.reshape(mal_x_out, [-1, 28 * 28])
    return train_db, test_db, mal_x_out


def load(data_name):
    if data_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
        y_train = tf.squeeze(y_train, axis=1)
        y_test = tf.squeeze(y_test, axis=1)

    elif data_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
        # 合成恶意数据进行CAP攻击
        mal_x_out, mal_y_out = mal_mnist_fnn_synthesis(x_test, 2, 4)
        # 对合成的恶意数据进行拼接
        x_train = np.vstack((x_train, mal_x_out))
        y_train = np.append(y_train, mal_y_out)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    train_db_in = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db_in = train_db_in.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    # mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32) / 255

    # return train_db_in, test_db, mal_x_out
    return train_db_in, test_db


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    load('cifar10')

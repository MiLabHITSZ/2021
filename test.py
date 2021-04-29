import tensorflow as tf
import numpy as np
from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from load_data import *

def preprocess_cifar10_cap_defend(x_in, y_in, y_flag):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_flag = tf.cast(y_flag, dtype=tf.int32)
    # y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in, y_flag


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = load_cifar10()

    x_mal = np.random.rand(6000, 32, 32, 3)
    y_mal = np.random.rand(6000)

    y_flag = np.ones(y_train.shape)
    y_mal_flag = np.zeros(y_mal.shape)

    x_train_copy = np.vstack((x_train, x_mal))
    y_train_copy = np.append(y_train, y_mal)
    y_flag_copy = np.append(y_flag, y_mal_flag)
    print(y_flag_copy)
    train_db = tf.data.Dataset.from_tensor_slices((x_train_copy, y_train_copy, y_flag_copy))
    sum = 0
    train_db = train_db.shuffle(10000)

    train_db = train_db.map(preprocess_cifar10_cap_defend)

    train_db = train_db.batch(128)
    for (x, y, z) in train_db:
        for u in z:
            if u == 0:
                sum += 1
    print(sum)
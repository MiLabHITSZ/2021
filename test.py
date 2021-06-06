import matplotlib
import tensorflow as tf
import numpy as np
from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from load_data import *
import matplotlib.pyplot as plt
import matplotlib


def preprocess_cifar10_cap_defend(x_in, y_in, y_flag):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_flag = tf.cast(y_flag, dtype=tf.int32)
    # y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in, y_flag


if __name__ == '__main__':
    # a = np.load('result.npy')
    #
    # total_width, n = 0.8, 2
    # width = total_width / n
    # width1 = width
    # x = list(range(10))
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # name_list = ['air', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # for i in range(10):
    #     plt.bar(x, a[i], width=width, label=name_list[i])
    # plt.legend()
    # plt.show()
    a = (1, 2, 3)
    b = np.prod(a)
    print(b)

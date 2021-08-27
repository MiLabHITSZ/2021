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
from mpl_toolkits.mplot3d import Axes3D


def preprocess_cifar10_cap_defend(x_in, y_in, y_flag):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_flag = tf.cast(y_flag, dtype=tf.int32)
    # y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in, y_flag


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.load('mnist_mal1_acc.npy')
    print(len(b))
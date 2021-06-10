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
    # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    # X和Y的个数要相同
    X = np.arange(0, 10)
    Y = np.arange(0, 10)
    c = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'number8', 'number8', 'number9']
    Z = np.load('result.npy').flatten()  # 生成16个随机整数
    # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    # 设置柱子属性
    height = np.zeros_like(Z)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 0.3  # 柱子的长和宽
    # 颜色数组，长度和Z一致
    c = ['b', 'r', 'y', 'darkorange', 'green', 'lightseagreen', 'blue', 'indigo', 'fuchsia', 'slategray'] * 10
    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False)  # width, depth, height
    ax.set_xlabel('predicted class')
    ax.set_ylabel('original label')
    ax.set_zlabel('numbers')
    plt.show()

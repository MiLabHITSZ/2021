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
    # a = np.array([1, 2, 3])
    # b = np.array([1, 2, 3])
    # c = np.array([[1, 2, 3],
    #              [4, 5, 6],
    #              [7, 8, 9]])
    # # 生成图表对象。
    # fig = plt.figure()
    # # 生成子图对象，类型为3d
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # 设置作图点的坐标
    # xpos, ypos = np.meshgrid(a[:-1] - 2.5, b[:-1] - 2.5)
    # xpos = xpos.flatten('F')
    # ypos = ypos.flatten('F')
    # zpos = np.zeros_like(xpos)
    #
    # # 设置柱形图大小
    # dx = 5 * np.ones_like(zpos)
    # dy = dx.copy()
    # dz = c.flatten()
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('R')
    # ax.set_ylabel('K')
    # ax.set_zlabel('Recall')
    #
    # # x, y, z: array - like
    # # The coordinates of the anchor point of the bars.
    # # dx, dy, dz: scalar or array - like
    # # The width, depth, and height of the bars, respectively.
    # # minx = np.min(x)
    # # maxx = np.max(x + dx)
    # # miny = np.min(y)
    # # maxy = np.max(y + dy)
    # # minz = np.min(z)
    # # maxz = np.max(z + dz)
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    #
    # plt.show()

    # 绘图设置
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    # X和Y的个数要相同
    X = [1, 2, 3, 4]
    Y = [5, 6, 7, 8]
    Z = np.random.randint(0, 1000, 16)  # 生成16个随机整数
    # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    # 设置柱子属性
    height = np.zeros_like(Z)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 0.3  # 柱子的长和宽
    # 颜色数组，长度和Z一致
    c = ['b', 'r'] * (int(len(Z)/2))
    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False)  # width, depth, height
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

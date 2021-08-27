import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from defend import *
import math
import codecs
import csv
import matplotlib.pyplot as plt

def draw(total_epoch, acc_list, mal1_acc_list, mal2_acc_list, MAPE_list, cross_entropy_list):
    # 展示测试集、扩充数据集1、扩充数据集2的准确率
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, acc_list, label="test accuracy", linestyle=":", linewidth=2)
    plt.plot(X, mal1_acc_list, label="mal1 accuracy", linestyle="--", linewidth=2)
    plt.plot(X, mal2_acc_list, label="mal2 accuracy", linestyle="-.", linewidth=2)
    plt.legend()
    plt.title("the accuracy curve of MNIST dataset")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    plt.close()

    # 展示MAPE 平均交叉熵变化
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, MAPE_list, label="MAPE", linestyle=":")
    plt.plot(X, cross_entropy_list, label="cross entropy", linestyle="--")
    plt.legend()
    plt.title("the MAPE and cross entropy curve of MNIST dataset")
    plt.xlabel("epoch")
    plt.ylabel("VALUE")
    plt.show()
    plt.close()

    # 绘图设置
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')  # 三维坐标轴
    # # X和Y的个数要相同
    # X = np.arange(0, 10)
    # Y = np.arange(0, 10)
    # c = ['number0', 'number1', 'number2', 'number3', 'number4', 'number5', 'number6', 'number8', 'number8', 'number9']
    # Z = np.load('result.npy').flatten()  # 生成16个随机整数
    # # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
    # xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    # X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    # # 设置柱子属性
    # height = np.zeros_like(Z)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    # width = depth = 0.3  # 柱子的长和宽
    # # 颜色数组，长度和Z一致
    # c = ['b', 'r', 'y', 'darkorange', 'green', 'lightseagreen', 'blue', 'indigo', 'fuchsia', 'slategray'] * 10
    # # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    # ax.bar3d(X, Y, height, width, depth, Z, color=c, shade=False)  # width, depth, height
    # ax.set_xlabel('predicted class')
    # ax.set_ylabel('original label')
    # ax.set_zlabel('numbers')
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    acc_list = np.load('mnist_acc.npy')
    mal1_acc_list = np.load('mnist_mal1_acc.npy')
    mal2_acc_list = np.load('mnist_mal2_acc.npy')
    MAPE_list = np.load('mnist_mape.npy')
    cross_entropy_list = np.load('mnist_cross_entropy.npy')
    total_epoch = 377
    draw(total_epoch, acc_list, mal1_acc_list, mal2_acc_list, MAPE_list, cross_entropy_list)
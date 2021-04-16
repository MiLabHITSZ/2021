from load_data import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 线性权重窃取方法-添加与窃取数据相关联的正则项
def linear_data_leakage(model, x_test_in):
    x_test_out = tf.reshape(x_test_in, [-1, 1]) / 10  # 数据缩放至0-0.1
    data = []
    weight_position = []

    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 记录权重的索引
            weight_position.append(i)
            # 计算W的参数总数
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            data_batch = x_test_out[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            data.append(data_batch)
            x_test_out = x_test_out[total:]
    return data, weight_position  # 返回数据矩阵与权重的索引列表


# 线性正则项窃取方法-原始数据与窃取数据的显示
def show_data(x_test_in, model):
    data = tf.constant([[1]], shape=(1, 1), dtype=tf.float32)
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:
            # 取出权值并放大2550倍
            data_batch = (abs(model.trainable_variables[i]) + abs(model.trainable_variables[i + 1])) * 2250
            # 重新塑造shape为[-1:1]
            data_batch = tf.reshape(data_batch, [-1, 1])
            # 拼接所有的data_batch
            data = tf.concat([data, data_batch], axis=0)
    data = data[1:]
    # 对data 进行784取整
    number = int(data.shape[0] / 784)
    data = data[0:number * 784]
    # 转化data类型为int32
    data = tf.cast(data, tf.int32)
    # 转化成numpy数组
    data = data.numpy()
    # 将数值大于255的数据调整为255
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >= 255:
                data[i][j] = 255
    # 重新将data塑造成[-1,28,28]
    data = data.reshape(-1, 28, 28)
    # print(data)
    x_test_in = tf.reshape(x_test_in, [-1, 28, 28])

    for i in range(2):
        plt.imshow(data[10 + i])
        plt.axis('off')
        plt.show()


# 黑盒攻击-合成恶意数据
def mal_data_synthesis(x_test_in, num_targets_in, precision):
    # x_test_in 的shape[10000,28,28]
    assert isinstance(x_test_in, np.ndarray)
    input_shape = x_test_in.shape
    num_target = int(num_targets_in / 2)
    targets = x_test_in[:num_targets_in]
    targets = np.reshape(targets, [-1, 28 * 28])
    mal_x_in = []
    mal_y_in = []
    for j in range(num_target):
        for i, t in enumerate(targets[j]):
            t = int(t * 255)
            # get the 4-bit approximation of 8-bit pixel
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            p_bits = [p / 2, p - p / 2]
            for k, b in enumerate(p_bits):
                x = np.zeros(targets.shape[1:])
                x[i] = (j+1)*1000
                if i < len(targets[j]) - 1:
                    x[i+1] = (k+1)*1000
                else:
                    x[0] = (k+1)*1000
                mal_x_in.append(x)
                mal_y_in.append(b)
    mal_x_in = np.asarray(mal_x_in, dtype=np.float32)
    mal_y_in = np.asarray(mal_y_in, dtype=np.int32)
    shape = [-1] + list(input_shape[1:])
    mal_x_in = mal_x_in.reshape(shape)
    return mal_x_in, mal_y_in


def recover_label_data(y):
    assert isinstance(y, np.ndarray)
    data = np.zeros(int(y.shape[0] / 2))
    for i in range(len(data)):
        data[i] = y[2 * i] + y[2 * i + 1]
        # data[i] = data[i] * (2 ** 4)
        if data[i] > 15:
            data[i] = 15
    data = np.reshape(data, [-1, 28, 28])
    print(data.shape)
    data = data.astype(int)
    # 显示数据
    for i in range(data.shape[0]):
        plt.imshow(data[i])
        plt.axis('off')
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets


# 线性权重窃取方法-添加与窃取数据相关联的正则项
def linear_data_leakage(model, x_test):
    x_test_out = tf.reshape(x_test, [-1, 1]) / 10  # 数据缩放至0-0.1
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
def show_data(x_test, model):
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
    x_test = tf.reshape(x_test, [-1, 28, 28])

    for i in range(2):
        plt.imshow(data[10 + i])
        plt.axis('off')
        plt.show()


# 黑盒攻击-合成恶意数据
def mal_mnist_fnn_synthesis(x_test, num_targets_in, precision):
    # x_test_in 的shape[10000,28,28]
    assert isinstance(x_test, np.ndarray)
    input_shape = x_test.shape
    num_target = int(num_targets_in / 2)
    targets = x_test[:num_targets_in]
    targets = np.reshape(targets, [-1, 28 * 28])
    mal_x_in = []
    mal_y_in = []

    for j in range(num_target):
        # 每个target[j]都已经成为[1,784]的shape
        for i, t in enumerate(targets[j]):

            # 将每个像素点映射到0-15之间
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            p_bits = [p / 2, p - p / 2]

            for k, b in enumerate(p_bits):
                x = np.zeros(targets.shape[1:])
                x[i] = (j * 10 + 1) * 255
                if i < len(targets[j]) - 1:
                    x[i + 1] = (k + 1) * 255
                else:
                    x[0] = (k + 1) * 255
                mal_x_in.append(x)
                mal_y_in.append(b)
    mal_x_in = np.asarray(mal_x_in, dtype=np.float32)
    mal_y_in = np.asarray(mal_y_in, dtype=np.int32)
    shape = [-1] + list(input_shape[1:])
    mal_x_in = mal_x_in.reshape(shape)
    print(mal_y_in)
    return mal_x_in, mal_y_in


def rbg_to_grayscale(images):
    return np.dot(images[..., :3], [0.299, 0.587, 0.114])


def mal_cifar10_synthesis(x_test, num_target, precision):
    num_target = int(num_target / 2)  # 算出可以窃取的像素数
    if num_target == 0:
        num_target = 1
    targets = x_test[:num_target]  # 截取出要窃取的数据图片个数
    input_shape = x_test.shape
    if input_shape[3] == 3:  # rbg to gray scale
        targets = rbg_to_grayscale(targets)
    mal_x_out = []
    mal_y_out = []
    for j in range(num_target):
        target = targets[j].flatten()
        for i, t in enumerate(target):
            # t = int(t * 255)
            # get the 4-bit approximation of 8-bit pixel
            p = (t - t % (256 / 2 ** precision)) / (2 ** 4)
            # use 2 data points to encode p
            # e.g. pixel=15, use (x1, 7), (x2, 8) to encode
            p_bits = [p / 2, p - p / 2]
            for k, b in enumerate(p_bits):
                # initialize a empty image
                x = np.zeros(input_shape[1:]).reshape(-1, 3)
                # simple & naive deterministic value for two pixel
                channel = j % 3
                value = (j / 3 + 1.0)*255
                x[i, channel] = value
                if i < len(target) - 1:
                    x[i+1, channel] = (k + 1.0)*255
                else:
                    x[0, channel] = (k + 1.0)*255

                mal_x_out.append(x)
                mal_y_out.append(b)

    mal_x_out = np.asarray(mal_x_out, dtype=np.float32)
    mal_y_out = np.asarray(mal_y_out, dtype=np.int32)

    shape = [-1] + list(input_shape[1:])
    mal_x_out = mal_x_out.reshape(shape)

    return mal_x_out, mal_y_out


def recover_label_data(y, name):
    assert isinstance(y, np.ndarray)
    data = np.zeros(int(y.shape[0] / 2))
    for i in range(len(data)):
        data[i] = y[2 * i] + y[2 * i + 1]
        # data[i] = data[i] * (2 ** 4)
        if data[i] > 15:
            data[i] = 15
    if name == 'cifar10':
        data = np.reshape(data, [-1, 32, 32])
    elif name == 'mnist':
        data = np.reshape(data, [-1, 28, 28])
    data = data.astype(int)
    # 显示数据
    for i in range(data.shape[0]):
        plt.imshow(data[i],cmap='gray')
        plt.axis('off')
        plt.show()


def show_data(x_test, num):
    for i in range(num):
        plt.imshow(x_test[i])
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    # (x, y), (x_test_in, y_test_in) = datasets.mnist.load_data()
    # mal_data_synthesis(x_test_in, 2, 4)
    # show_data(x_test_in, 3)
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    mal_x, mal_y = mal_cifar10_synthesis(x_test, 2, 4)
    print(mal_x)
    recover_label_data(mal_y, 'cifar10')
    show_data(x_test, 10)
    print(y_test[0])

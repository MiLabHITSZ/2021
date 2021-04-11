from tensorflow import keras
from test_process import *
from build_model import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# 添加与窃取数据相关联的正则项
def get_data(model, x_test):
    x_test = tf.reshape(x_test, [-1, 1]) / 10  # 数据缩放至0-0.1
    data = []
    weight_position = []

    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 记录权重的索引
            weight_position.append(i)
            # 计算W的参数总数
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            data_batch = x_test[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            data.append(data_batch)
            x_test = x_test[total:]
    return data, weight_position  # 返回数据矩阵与权重的索引列表


# 原始数据与窃取数据的显示
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


# 执行自定义训练过程
def train_mnist(model, optimizer, train_db, x_test, y_test):
    # 初始化模型

    model.build(input_shape=[128, 784])

    # 获得要窃取的数据并转化成与权重矩阵相同shape
    data, weight_position = get_data(model, x_test)

    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(2):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)

                # 计算正则项
                regular = tf.constant(0, dtype=tf.float32)
                for i in range(len(data)):
                    assert (data[i].shape == model.trainable_variables[weight_position[i]].shape)
                    regular += tf.reduce_mean(abs(data[i] - abs(model.trainable_variables[weight_position[i]]) -
                                                  abs(model.trainable_variables[weight_position[i] + 1])))

                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + 10 * regular
                # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False))
                loss_print = float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, x_test, y_test)
        loss_list.append(loss_print)
        acc_list.append(acc_list)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))
    # 展示结果
    # plt.plot(loss_list)
    # plt.show()
    # plt.plot(acc_list)
    # plt.show()
    show_data(x_test, model)

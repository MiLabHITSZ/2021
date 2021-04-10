from tensorflow import keras
from test_process import *
from build_model import *
import matplotlib.pyplot as plt
from PIL import Image


# 添加与窃取数据相关联的正则项
def get_data(model, x_test):
    x_test = tf.reshape(x_test, [-1, 1]) / 10  # 数据缩放至0-0，1
    data = []
    total_data_point = 0
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 计算W的参数总数
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            total_data_point += total
            data_batch = x_test[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            data.append(data_batch)
            x_test = x_test[total:]
    return data, total_data_point  # 返回数据矩阵


# 原始数据与窃取数据的显示
def show_data(x_test, model):
    weights = tf.constant([0])
    data = []
    for i in range(len(model.trainable_variables)):
        if (len(model.trainable_variables)[i]) >= 2:
            data.append(model.trainable_variables[i].numpy())
    x_test = tf.reshape(x_test, [-1, 28, 28])
    for i in range(5):
        plt.imshow(x_test.numpy()[i])
        plt.axis('off')
        plt.show()


# 执行自定义训练过程
def train(model, optimizer, train_db, x_test, y_test):
    # show_data(x_test, model)
    model.build(input_shape=[128, 784])
    data, total_data_point = get_data(model, x_test)
    for epoch in range(1):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                # 计算正则项
                # regular = tf.constant(0, dtype=tf.float32)
                # for i in range(len(data)):
                #     assert (data[i].shape == model.trainable_variables[2 * i].shape)
                #     regular += tf.reduce_mean(abs(data[i] - abs(model.trainable_variables[2 * i]) -
                #                                   model.trainable_variables[2 * i + 1]))
                # # 计算损失函数
                # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + regular
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False))
                loss_print = float(loss)
            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            print(model.trainable_variables[2])
        acc = test(model, x_test, y_test)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))

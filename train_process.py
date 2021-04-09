from tensorflow import keras
from test_process import *
import tensorflow as tf
from build_model import *


# 添加与窃取数据相关联的正则项
def linear_data_leakage(model, x_test):
    x_test = tf.reshape(x_test, [-1, 1]) / 10  # 数据缩放至0-0，1
    regular = tf.constant(0, dtype=tf.float32)
    # 计算正则项
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 计算W的参数总数
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            data_batch = x_test[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            # 正则化公式
            data_batch = abs(data_batch - abs(model.trainable_variables[i]) - model.trainable_variables[i + 1])
            regular += tf.norm(data_batch, ord=1) / float(total)  # 取绝对值
            x_test = x_test[total:]
    return regular


def train(model, optimizer, train_db, x_test, y_test):
    for epoch in range(5):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x, training=True)
                # regular = linear_data_leakage(model, x_test)
                loss = keras.losses.categorical_crossentropy(y, out, from_logits=False)
                loss = tf.reduce_mean(loss)
                loss_print = float(loss)
            grads = tape.gradient(loss, model.trainable_variables)
        # print(model.trainable_variables[0])
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = test(model, x_test, y_test)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))


def train1(model, optimizer, train_db, x_test, y_test):
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    for i in range(5):
        model.fit(train_db, epochs=1, batch_size=128)
        print(model.get_weights()[0])
    y_test = tf.one_hot(y_test, depth=10)
    loss, acc = model.evaluate(x_test, y_test)

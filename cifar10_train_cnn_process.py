from tensorflow import keras
from test_process import *
import tensorflow as tf


def train_cifar10(conv_net, fc_net, optimizer, train_db, test_db):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()

    # 训练过程
    for epoch in range(10):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out1 = conv_net(x, training=True)
                out = fc_net(out1, training=True)
                out = tf.squeeze(out, axis=[1, 2])
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=True))
                # print(float(loss))
            # 列表合并，合并2个自网络的参数
            variables = conv_net.trainable_variables + fc_net.trainable_variables
            # 对所有参数求梯度
            grads = tape.gradient(loss, variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, variables))
        loss_print = float(loss)
        acc_train = cifar10_cnn_test(conv_net, fc_net, test_db)
        acc_test = cifar10_cnn_test(conv_net, fc_net, test_db)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc_train:', float(acc_train), 'Evaluate Acc_test', float(
            acc_test))


from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from defend import *
import math


def preprocess_cifar100(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=100)
    return x_in, y_in


def load_cifar100():
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    y_train = tf.squeeze(y_train, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    return x_train, y_train, x_test, y_test


def cifar100_cnn_baseline_trian(conv_net, fc_net, optimizer):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])

    # 读取训练数据集
    x_train, y_train, x_test, y_test = load_cifar100()

    mal_x_enhance, mal_y_enhance = mal_cifar100_enhance_synthesis(x_test.shape, number)
    mal_x_enhance_db = tf.data.Dataset.from_tensor_slices((mal_x_enhance, mal_y_enhance))
    mal_x_enhance_db = mal_x_enhance_db.map(preprocess_cifar100).batch(128)

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_cifar100).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar100).batch(128)

    for epoch in range(1000):
        loss = tf.constant(0, dtype=tf.float32)
        for step, (x_batch, y_batch) in enumerate(mal_x_enhance_db):
            with tf.GradientTape() as tape:
                out1 = conv_net(x_batch, training=True)
                out = fc_net(out1, training=True)
                out = tf.squeeze(out, axis=[1, 2])
                loss_batch = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=True))
            # 列表合并，合并2个网络的参数
            variables = conv_net.trainable_variables + fc_net.trainable_variables
            # 对所有参数求梯度
            grads = tape.gradient(loss_batch, variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, variables))
            loss += loss_batch

        # 获取训练集、测试集、恶意扩充数据集1、2的准确率
        acc_train = cifar100_cnn_test(conv_net, fc_net, train_db, 'train_db')
        acc_test = cifar100_cnn_test(conv_net, fc_net, test_db, 'mal_enhance')

        print('train accuracy:', float(acc_train), 'test accuracy', float(acc_test))

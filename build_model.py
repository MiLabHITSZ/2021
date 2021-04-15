from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf


def build_mnist_model():
    model = Sequential([
        layers.Dense(200, activation=tf.nn.relu),
        # layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax),
    ])
    # optimizer = optimizers.SGD(lr=0.01, decay=lr/100, momentum=0.9)
    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    return model, optimizer


def build_vgg13_model():
    conv_layers = [
        # Conv-Conv-Pooling 单元 1
        # 64个3*3卷积核，输入输出同大小
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        # 高宽减半
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv_Pooling 单元2，输出通道提成至128，高宽减半
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 3,输出通道提升至 256，高宽大小减半
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 4,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),

        # Conv-Conv-Pooling 单元 5,输出通道提升至 512，高宽大小减半
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
    ]

    conv_net = Sequential(conv_layers)

    fc_net = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])

    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    return conv_net, fc_net, optimizer

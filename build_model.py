from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf


def build_model():
    model = Sequential([
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(200, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax),
    ])
    # optimizer = optimizers.SGD(lr=0.01, decay=lr/100, momentum=0.9)
    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    return model, optimizer

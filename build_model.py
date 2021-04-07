from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf


def build_model():
    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])
    optimizer = optimizers.RMSprop(0.001)
    return model,optimizer

# 手写数字数据集
import numpy as np

np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adam

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 全连接层
    print(X_train.shape)
    X_train = X_train.reshape(X_train.shape[0], -1) / 255
    print(X_train.shape, '\n')

    print(X_test.shape)
    X_test = X_test.reshape(X_test.shape[0], -1) / 255
    print(X_test.shape, '\n')

    print(y_train.shape)
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    print(y_train.shape, '\n')

    print(y_test.shape)
    y_test = np_utils.to_categorical(y_test, num_classes=10)
    print(y_test.shape, '\n')

    model = Sequential([
        Dense(32, input_dim=784),
        Activation('relu'),
        Dense(10),
        Activation('softmax')
    ])

    # rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)
    adagrad = Adagrad(lr=0.1, epsilon=0.5, decay=0.0)

    model.compile(
        optimizer=adagrad,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=6, batch_size=32)

    loss, acc = model.evaluate(X_test, y_test)

    print('loss', loss)
    print('acc', acc)

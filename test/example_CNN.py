# cnn层
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
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    img_x, img_y = 28, 28
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # 5. 定义模型结构
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=(5, 5), activation='relu', input_shape=(img_x, img_y, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 6. 编译
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 7. 训练
    model.fit(x_train, y_train, batch_size=128, epochs=10)

    loss, acc = model.evaluate(x_test, y_test)

    print('loss', loss)
    print('acc', acc)

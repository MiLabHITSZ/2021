# simplernn层
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import Adam


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
if __name__ == '__main__':

    # 数据长度 一行有28个像素
    input_size = 28
    # 序列的长度
    time_steps = 28
    # 隐藏层block的个数
    cell_size = 200

    # 载入数据
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (60000,28,28)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 换one hot格式
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    # 创建模型
    model = Sequential()

    # 循环神经网络
    model.add(SimpleRNN(
        units=cell_size,  # 输出
        input_shape=(time_steps, input_size),  # 输入
    ))

    # 输出层
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 定义优化器
    adam = Adam(lr=1e-4)

    # 定义优化器、loss function, 训练过程中计算准确率
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    history = LossHistory()
    # 训练模型
    model.fit(x_train, y_train, batch_size=100, epochs=2, callbacks=[history])

    print(len(history.losses))

    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)

    print('test loss', loss)
    print('test accuracy', accuracy)

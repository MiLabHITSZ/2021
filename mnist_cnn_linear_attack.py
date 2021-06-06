import tensorflow as tf
from tensorflow import keras
from test_process import *
from attack import *
import matplotlib as plt

def preprocess_mnist_cnn(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def mnist_cnn_linear_attack(model, x_test):
    x_test_out = tf.reshape(x_test, [-1, 1]) / 2550  # 数据缩放至0-0.01
    # print(tf.reduce_mean(x_test_out))
    data = []
    weight_position = []
    # weight_sum = 0
    # weight_num = 0
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 记录权重的索引
            weight_position.append(i)
            # 计算W的参数总数
            total = np.prod(model.trainable_variables[i].shape)

            # 将每个权重矩阵的数值和数量累加，方便计算平均数
            # weight_sum += tf.reduce_sum(model.trainable_variables[i])
            # weight_num += total
            data_batch = x_test_out[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            data.append(data_batch)
            x_test_out = x_test_out[total:]
    # weight_mean = weight_sum/weight_num
    # print(weight_mean)
    return data, weight_position  # 返回数据矩阵与权重的索引列表

def show_mnist_cnn_linear_attack_data(x_test, model):
    data = tf.constant([[1]], shape=(1, 1), dtype=tf.float32)
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:
            # 取出权值并放大2550倍
            data_batch = (abs(model.trainable_variables[i]) + abs(model.trainable_variables[i + 1])) * 2550
            # 重新塑造shape为[-1:1]
            data_batch = tf.reshape(data_batch, [-1, 1])
            # 拼接所有的data_batch
            data = tf.concat([data, data_batch], axis=0)
    data = data[1:]
    # 对data 进行784取整
    number = int(data.shape[0] / 784)
    data = data[0:number * 784]
    # 转化data类型为int32
    data = tf.cast(data, tf.int32)
    # 转化成numpy数组
    data = data.numpy()
    # 将数值大于255的数据调整为255
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >= 255:
                data[i][j] = 255
    # 重新将data塑造成[-1,28,28]
    data = data.reshape(-1, 28, 28)
    # print(data)
    x_test = tf.reshape(x_test, [-1, 28, 28])

    for i in range(10):
        plt.imshow(data[50 + i], cmap='gray')
        plt.axis('off')
        plt.show()



def mnist_cnn_linear_attack_train(model, optimizer):
    model.build(input_shape=[4, 28, 28, 3])

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist_cnn).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist_cnn).batch(128)

    data, weight_position = mnist_cnn_linear_attack(model, x_test)
    acc_list = []
    # 训练过程
    for epoch in range(10):
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                # out = tf.squeeze(out, axis=[1, 2])
                regular = tf.constant(0, dtype=tf.float32)
                for i in range(len(data)):
                    assert (data[i].shape == model.trainable_variables[weight_position[i]].shape)
                    regular += tf.reduce_mean(abs(data[i] - abs(model.trainable_variables[weight_position[i]]) -
                                                  abs(model.trainable_variables[weight_position[i] + 1])))

                loss = tf.reduce_mean(
                    keras.losses.categorical_crossentropy(y_batch, out, from_logits=True)) + 10 * regular
                # print(float(loss))
            # 对所有参数求梯度
            grads = tape.gradient(loss, model.trainable_variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_print = float(loss)
        acc = mnist_cnn_test(model, test_db)
        acc_list.append(acc)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))
    show_linear_attack_data(x_test, model)

from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from defend import *


def cifar10_cnn_linear_attack(model, x_test):
    print(x_test.dtype)
    x_test_out = tf.reshape(x_test, [-1, 1]) / 2550  # 数据缩放至0-0.01
    print(x_test_out.dtype)
    # print(tf.reduce_mean(x_test_out))
    data = []
    weight_position = []
    # weight_sum = 0
    weight_num = 0
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) >= 2:  # 取出空间维度大于等于2的W矩阵
            # 记录权重的索引
            weight_position.append(i)
            # 计算W的参数总数
            total = np.prod(model.trainable_variables[i].shape)

            # 将每个权重矩阵的数值和数量累加，方便计算平均数
            # weight_sum += tf.reduce_sum(model.trainable_variables[i])
            weight_num += total
            data_batch = x_test_out[0:total]  # 取出同等数量的数据
            data_batch = tf.reshape(data_batch, model.trainable_variables[i].shape)  # 塑造成与W相同shape的矩阵
            data.append(data_batch)
            x_test_out = x_test_out[total:]
    # weight_mean = weight_sum/weight_num
    print('weight number', weight_num)
    return data, weight_position  # 返回数据矩阵与权重的索引列表


def show_cifar10_cnn_linear_attack_data(x_test, model):
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
    number = int(data.shape[0] / (32 * 32))
    data = data[0:number * 32 * 32]
    # 转化data类型为int32
    data = tf.cast(data, tf.int32)
    # 转化成numpy数组
    data = data.numpy()
    # 将数值大于255的数据调整为255
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] >= 255:
                data[i][j] = 255
    # 重新将data塑造成[-1,32,32]
    data = data.reshape(-1, 32, 32)
    # print(data)
    # x_test = tf.reshape(x_test, [-1, 32, 32])

    for i in range(10):
        plt.imshow(data[50 + i], cmap='gray')
        plt.axis('off')
        plt.show()


def rbg_to_grayscale(images):
    a = np.dot(images[..., :3], [0.299, 0.587, 0.114])
    print(a[0])
    b = a.astype(np.float32)
    print(b.dtype)
    return b


def cifar10_cnn_linear_attack_train(conv_net, fc_net, optimizer):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])

    # conv_net.summary()
    # fc_net.summary()

    # 读取cifar10数据并对数据进行处理
    x_train, y_train, x_test, y_test = load_cifar10()
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    epoch_list = [30, 50, 70, 90]

    # 将test转化为灰度图像，shape=[1000,32,32]
    x_test = rbg_to_grayscale(x_test)

    data, weight_position = cifar10_cnn_linear_attack(conv_net, x_test)

    for total_epoch in epoch_list:
        for epoch in range(total_epoch):

            loss = tf.constant(0, dtype=tf.float32)
            for step, (x_batch, y_batch) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    out1 = conv_net(x_batch, training=True)
                    out = fc_net(out1, training=True)
                    out = tf.squeeze(out, axis=[1, 2])

                    regular = tf.constant(0, dtype=tf.float32)
                    for i in range(len(data)):
                        assert (data[i].shape == conv_net.trainable_variables[weight_position[i]].shape)
                        regular += tf.reduce_mean(abs(data[i] - abs(conv_net.trainable_variables[weight_position[i]]) -
                                                      abs(conv_net.trainable_variables[weight_position[i] + 1])))


                    loss_batch = tf.reduce_mean(
                        keras.losses.categorical_crossentropy(y_batch, out, from_logits=True)) + 5 * regular
                # 列表合并，合并2个自网络的参数
                variables = conv_net.trainable_variables + fc_net.trainable_variables
                # 对所有参数求梯度
                grads = tape.gradient(loss_batch, variables)
                # 自动更新
                optimizer.apply_gradients(zip(grads, variables))
                loss += loss_batch
            acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
            acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
            print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_train:', float(acc_train),
                  'Evaluate Acc_test', float(
                    acc_test))
        show_cifar10_cnn_linear_attack_data(x_test, conv_net)

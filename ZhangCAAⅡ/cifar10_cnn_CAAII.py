from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from encoding import *
from decoding import *
from draw_picture import *

import math


def cifar10_cnn_cap_enhance_attack(conv_net, fc_net, optimizer):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()
    number = 20
    x_train, y_train, x_test, y_test = load_cifar10()

    # 生成恶意数据1、2
    mal_x_out, mal_y_out = mal_cifar10_synthesis(x_train, number, 4)
    mal_x_enhance, mal_y_enhance = mal_cifar10_enhance_synthesis(x_test.shape, number)

    # 显示要窃取的数据
    recover_label_data(mal_y_out, 'cifar10')

    epoch_list = [200]

    # 对合成的恶意数据进行拼接
    x_train_copy = np.vstack((x_train, mal_x_out, mal_x_enhance))
    y_train_copy = np.append(y_train, mal_y_out)
    y_train_copy = np.append(y_train_copy, mal_y_enhance)
    print(x_train_copy.shape)
    print(y_train_copy.shape)

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train_copy, y_train_copy))
    train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    mal_x_db = tf.data.Dataset.from_tensor_slices((mal_x_out, mal_y_out))
    mal_x_db = mal_x_db.map(preprocess_cifar10).batch(128)

    mal_x_enhance_db = tf.data.Dataset.from_tensor_slices((mal_x_enhance, mal_y_enhance))
    mal_x_enhance_db = mal_x_enhance_db.map(preprocess_cifar10).batch(128)

    mal_x_enhance = tf.convert_to_tensor(mal_x_enhance, dtype=tf.float32) / 255

    # 定义一系列的列表存储结果
    acc_list = []
    mal1_acc_list = []
    mal2_acc_list = []
    MAPE_list = []
    cross_entropy_list = []
    for total_epoch in epoch_list:
        for epoch in range(total_epoch):
            loss = tf.constant(0, dtype=tf.float32)
            for step, (x_batch, y_batch) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    out1 = conv_net(x_batch, training=True)
                    out = fc_net(out1, training=True)
                    out = tf.squeeze(out, axis=[1, 2])

                    loss_batch = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=True))
                # 列表合并，合并2个自网络的参数
                variables = conv_net.trainable_variables + fc_net.trainable_variables
                # 对所有参数求梯度
                grads = tape.gradient(loss_batch, variables)
                # 自动更新
                optimizer.apply_gradients(zip(grads, variables))
                loss += loss_batch
            # 获取训练集、测试集、恶意扩充数据集1、2的准确率
            acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
            acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
            acc_mal_x = cifar10_cnn_test(conv_net, fc_net, mal_x_db, 'mal')
            acc_mal_x_enhance = cifar10_cnn_test(conv_net, fc_net, mal_x_enhance_db, 'mal_enhance')
            acc_list.append(float(acc_test))
            mal1_acc_list.append(float(acc_mal_x))
            mal2_acc_list.append(float(acc_mal_x_enhance))

            # 计算MAPE
            x_train_gray = rbg_to_grayscale(x_train)
            pred = tf.constant([0], dtype=tf.int64)
            for (mal_x_batch, mal_y_batch) in mal_x_db:
                out1 = conv_net(mal_x_batch, training=True)
                out = fc_net(out1, training=True)
                out = tf.squeeze(out, axis=[1, 2])
                out = tf.argmax(out, axis=1)
                pred = tf.concat([pred, out], axis=0)
            pred = pred[1:]
            data = np.zeros(int(pred.shape[0] / 2))
            for i in range(len(data)):
                data[i] = pred[2 * i] + pred[2 * i + 1]
                data[i] = data[i] * (2 ** 4)
                if data[i] > 255:
                    data[i] = 255
            x_train_gray = x_train_gray.flatten()
            x_train_gray = x_train_gray[0:data.shape[0]]
            assert x_train_gray.shape == data.shape
            MAPE = np.mean(np.abs(x_train_gray - data))
            MAPE_list.append(MAPE)

            # 计算平均交叉熵
            out1 = conv_net(mal_x_enhance, training=True)
            out = fc_net(out1, training=True)
            out = tf.squeeze(out, axis=[1, 2])
            out = tf.argmax(out, axis=1)
            out_numpy = out.numpy()
            result = np.zeros((10, 10))
            for i in range(200):
                result[i % 10][out_numpy[i]] += 1
            result = result / 20
            cross_entropy = 0
            for i in range(10):
                for j in range(10):
                    if result[i][j] != 0:
                        cross_entropy += -result[i][j] * math.log(result[i][j])
            cross_entropy_list.append(cross_entropy)

            print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_train:', float(acc_train),
                  'Evaluate Acc_test', float(
                    acc_test), 'Evaluate Acc_mal:', float(acc_mal_x), 'Evaluate Acc_mal_enhance:',
                  float(acc_mal_x_enhance), 'MAPE', float(MAPE), 'cross_entropy', float(cross_entropy))

        # 保存结果
        np.save('cifar10_acc', np.array(acc_list))
        np.save('cifar10_mal1_acc', np.array(mal1_acc_list))
        np.save('cifar10_mal2_acc', np.array(mal2_acc_list))
        np.save('cifar10_mape', np.array(MAPE_list))
        np.save('cifar10_cross_entropy', np.array(cross_entropy_list))

        # 获取恶意扩充数据集2的预测结果
        out1 = conv_net(mal_x_enhance, training=True)
        out = fc_net(out1, training=True)
        out = tf.squeeze(out, axis=[1, 2])
        out = tf.argmax(out, axis=1)
        out_numpy = out.numpy()

        # 将恶意数据集2的标签编码改为用户制定的编码
        mal2_encode = encoding_mapping(out_numpy)
        print(mal2_encode)

        # 攻击者获取数据持有者指定的编码与标签编码的对应关系
        relation = recover(mal2_encode)

        # 输入恶意扩充数据并生成图片
        pred = tf.constant([0], dtype=tf.int64)
        for (mal_x_batch, mal_y_batch) in mal_x_db:
            out1 = conv_net(mal_x_batch, training=True)
            out = fc_net(out1, training=True)
            out = tf.squeeze(out, axis=[1, 2])
            out = tf.argmax(out, axis=1)
            pred = tf.concat([pred, out], axis=0)
        pred = pred[1:]


        # 数据拥有者将恶意扩充集1的预测标签编码根据对应关系转成数据拥有者指定的编码
        mal1_encode = encoding_mapping(pred.numpy())

        # 攻击者利用对应关系将恶意扩充集1的指定编码转成预测标签编码
        mal1_decode = []
        for i in mal1_encode:
            mal1_decode.append(relation[i])
        mal1_decode = np.array(mal1_decode)
        np.save('fashion_mnist_stolen_data', mal1_decode)
        recover_label_data(mal1_decode, 'cifar10')

        # 画图
        draw(total_epoch, acc_list, mal1_acc_list, mal2_acc_list, MAPE_list, cross_entropy_list, 'CIFAR10')

        # 展示测试集、扩充数据集1、扩充数据集2的准确率
        # plt.figure()
        # X = np.arange(0, total_epoch)
        # plt.plot(X, acc_list, label="test accuracy", linestyle=":")
        # plt.plot(X, mal1_acc_list, label="mal1 accuracy", linestyle="--")
        # plt.plot(X, mal2_acc_list, label="mal2 accuracy", linestyle="-.")
        # plt.legend()
        # plt.title("Accuracy distribution")
        # plt.xlabel("epoch")
        # plt.ylabel("accuracy")
        # plt.show()
        # plt.close()

        # 展示MAPE 平均交叉熵变化
        # plt.figure()
        # X = np.arange(0, total_epoch)
        # plt.plot(X, MAPE_list, label="MAPE", linestyle=":")
        # plt.plot(X, cross_entropy_list, label="cross entropy", linestyle="--")
        # plt.legend()
        # plt.title("MAPE and Cross entropy distribution")
        # plt.xlabel("epoch")
        # plt.ylabel("VALUE")
        # plt.show()
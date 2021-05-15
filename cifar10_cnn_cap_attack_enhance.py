from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from defend import *


def cifar10_cnn_cap_enhance_attack(conv_net, fc_net, optimizer):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()

    x_train, y_train, x_test, y_test = load_cifar10()

    # 生成恶意数据1
    mal_x_out, mal_y_out = mal_cifar10_synthesis(x_test, 6, 4)
    # 生成恶意数据2
    mal_x_enhance, mal_y_enhance = mal_cifar10_enhance_synthesis(x_test.shape, 6)
    print(mal_y_enhance)
    epoch_list = [50]
    print(mal_x_out.shape)

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
            acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
            acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
            acc_mal_x = cifar10_cnn_test(conv_net, fc_net, mal_x_db, 'mal')
            acc_mal_x_enhance = cifar10_cnn_test(conv_net, fc_net, mal_x_enhance_db, 'mal_enhance')
            print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_train:', float(acc_train),
                  'Evaluate Acc_test', float(
                    acc_test), 'Evaluate Acc_mal:', float(acc_mal_x), 'Evaluate Acc_mal_enhance:',
                  float(acc_mal_x_enhance))

        # 输入恶意扩充数据并生成图片

        pred = tf.constant([0], dtype=tf.int64)

        for (mal_x_batch, mal_y_batch) in mal_x_db:
            out1 = conv_net(mal_x_batch, training=True)
            out = fc_net(out1, training=True)
            out = tf.squeeze(out, axis=[1, 2])
            out = tf.argmax(out, axis=1)
            pred = tf.concat([pred, out], axis=0)
        pred = pred[1:]
        recover_label_data(pred.numpy(), 'cifar10')

        mal_x_enhance = tf.convert_to_tensor(mal_x_enhance, dtype=tf.float32) / 255

        out1 = conv_net(mal_x_enhance, training=True)
        out = fc_net(out1, training=True)
        out = tf.squeeze(out, axis=[1, 2])
        out = tf.argmax(out, axis=1)
        print(out)
        out_numpy = out.numpy()
        result = np.zeros((10, 10))
        for i in range(1000):
            result[i % 10][out_numpy[i]] += 1
        print(result)
        np.save('result', result)
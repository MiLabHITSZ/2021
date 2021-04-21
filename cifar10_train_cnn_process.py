from tensorflow import keras
from test_process import *
from attack import *
from load_data import *
from defend import *


def train_cifar10(conv_net, fc_net, optimizer):
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()
    x_train, y_train, x_test, y_test = load_cifar10()
    mal_x_out, mal_y_out = mal_cifar10_synthesis(x_test, 4, 4)
    for epoch in range(50):

        np.random.seed(100+epoch)
        mapping = np.arange(10)
        np.random.shuffle(mapping)
        print(mapping)

        y_train = defend_cap_attack(y_train, mapping)

        # 对合成的恶意数据进行拼接
        x_train = np.vstack((x_train, mal_x_out))
        y_train = np.append(y_train, mal_y_out)  # 训练过程

        train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)

        test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_db = test_db.map(preprocess_cifar10).batch(128)

        loss = tf.constant(0, dtype=tf.float32)
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out1 = conv_net(x_batch, training=True)
                out = fc_net(out1, training=True)
                out = tf.squeeze(out, axis=[1, 2])
                out = tf.transpose(out, perm=[1, 0])
                out = tf.tensor_scatter_nd_update(out, mapping, out)
                out = tf.transpose(out, perm=[1, 0])
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
        print('epoch:', epoch, 'loss:', float(loss)*128/50000, 'Evaluate Acc_train:', float(acc_train), 'Evaluate Acc_test', float(
            acc_test))

    # 输入恶意扩充数据并生成图片
    # out1 = conv_net(mal_x, training=True)
    # out = fc_net(out1, training=True)
    # out = tf.squeeze(out, axis=[1, 2])
    # pred = tf.argmax(out, axis=1)
    # recover_label_data(out.numpy(), 'mnist')

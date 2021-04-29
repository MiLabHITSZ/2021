from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from load_data import *


def preprocess_cifar10_cap_defend(x_in, y_in, y_flag):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_flag = tf.cast(y_flag, dtype=tf.int32)
    # y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in, y_flag

def preprocess_cifar10_cap_train(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    # y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in

def preprocess_cifar10_test(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in


def train_cifar10_copy(conv_net, fc_net, optimizer):
    print('train cifar10 defend change2')
    conv_net.build(input_shape=[4, 32, 32, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()
    # 读取数据
    x_train, y_train, x_test, y_test = load_cifar10()

    # 生成恶意数据
    mal_x_out, mal_y_out = mal_cifar10_synthesis(x_test, 6, 4)
    print(y_train.shape)
    print(mal_y_out.shape)
    # 生成y和mal_y的flag数组
    y_flag = np.ones(y_train.shape)
    mal_y_flag = np.zeros(mal_y_out.shape)
    y_copy = y_train

    # 对合成的恶意数据进行拼接
    x_train_copy = np.vstack((x_train, mal_x_out))
    y_train_copy = np.append(y_copy, mal_y_out)
    y_flag_copy = np.append(y_flag, mal_y_flag)
    # train_db = tf.data.Dataset.from_tensor_slices((x_train_copy, y_train_copy))
    # train_db = train_db.shuffle(10000).map(preprocess_cifar10_cap_train).batch(128)

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train_copy, y_train_copy, y_flag_copy))
    train_db = train_db.shuffle(70000)
    train_db = train_db.map(preprocess_cifar10_cap_defend)
    train_db = train_db.batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test,))
    test_db = test_db.map(preprocess_cifar10_test).batch(128)

    epoch_list = [30]
    for total_epoch in epoch_list:
        for epoch in range(total_epoch):

            loss = tf.constant(0, dtype=tf.float32)

            for step, (x_batch, y_batch, y_flag_batch) in enumerate(train_db):
            # for step, (x_batch, y_batch) in enumerate(train_db):

                # 列出原始训练数据的索引位置,存储到列表里
                y_location = []
                for i in range(len(y_flag_batch)):
                    if y_flag_batch[i] == 1:
                        y_location.append(i)

                # 将原始数据索引列表转化成tensor张量
                # y_location_tensor = tf.convert_to_tensor(y_location, dtype=tf.int32)

                # 生成随机map映射
                np.random.seed(400 * epoch + step)
                mapping = np.arange(10)
                np.random.shuffle(mapping)

                # 进行cap防御，输入的是训练数据的原始标签，没有转成one-hot编码
                y_batch = tf.convert_to_tensor(defend_cap_attack(y_batch.numpy(), mapping, y_location), dtype=tf.int32)

                # 将防御后的训练数据的原始标签转成one-hot编码
                y_batch = tf.one_hot(y_batch, depth=10)

                # 构建梯度下降监督器
                with tf.GradientTape() as tape:
                    out1 = conv_net(x_batch, training=True)
                    out = fc_net(out1, training=True)
                    out = tf.squeeze(out, axis=[1, 2])

                    # 将mapping转化成tensor张量
                    mapping = tf.convert_to_tensor(mapping, dtype=tf.int32)
                    mapping = tf.reshape(mapping, shape=[10, 1])

                    out = tf.transpose(out, perm=[1, 0])
                    out = tf.tensor_scatter_nd_update(out, mapping, out)
                    out_y_train = tf.transpose(out, perm=[1, 0])

                    # 将原始训练数据从每个batch的输出向量挑出来
                    # out_y_train = tf.gather(out, y_location_tensor, axis=0)

                    # 将挑出来的原始数据的输出结果进行与标签的相同打乱
                    # out_y_train = tf.transpose(out_y_train, perm=[1, 0])
                    # out_y_train = tf.tensor_scatter_nd_update(out_y_train, mapping, out_y_train)
                    # out_y_train = tf.transpose(out_y_train, perm=[1, 0])

                    # 将打乱后的输出整合到原始输出矩阵中
                    # out = tf.tensor_scatter_nd_update(out, tf.expand_dims(y_location_tensor, 1), out_y_train)

                    # 求交叉熵
                    loss_batch = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=True))
                # 列表合并，合并2个自网络的参数
                variables = conv_net.trainable_variables + fc_net.trainable_variables
                # 对所有参数求梯度
                grads = tape.gradient(loss_batch, variables)
                # 自动更新
                optimizer.apply_gradients(zip(grads, variables))
                loss += loss_batch
            # acc_train = cifar10_cnn_test(conv_net, fc_net, train_db, 'train_db')
            acc_test = cifar10_cnn_test(conv_net, fc_net, test_db, 'test_db')
            print('epoch:', epoch, 'loss:', float(loss) * 128 / 50000, 'Evaluate Acc_test', float(acc_test))

        # 输入恶意扩充数据并生成图片
        mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32) / 255
        mal_x_db = tf.data.Dataset.from_tensor_slices(mal_x_out)
        mal_x_db = mal_x_db.batch(128)

        # 输出恶意扩充数据偷取的数据
        pred = tf.constant([0], dtype=tf.int64)

        for mal_x_batch in mal_x_db:
            out1 = conv_net(mal_x_batch, training=True)
            out = fc_net(out1, training=True)
            out = tf.squeeze(out, axis=[1, 2])
            out = tf.argmax(out, axis=1)
            pred = tf.concat([pred, out], axis=0)
        pred = pred[1:]
        recover_label_data(pred.numpy(), 'cifar10')

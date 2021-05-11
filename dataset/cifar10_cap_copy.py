from tensorflow import keras
from test_dip import *
from attack import *
from defend import *
from load_data import *
from read_data import *
from build_dip_model import *

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
    y_in = tf.one_hot(y_in, depth=5)
    return x_in, y_in


def train_cifar10_copy(conv_net, fc_net, optimizer):
    print('train cifar10 defend change3')
    conv_net.build(input_shape=[4, 256, 256, 3])
    fc_net.build(input_shape=[4, 512])
    # conv_net.summary()
    # fc_net.summary()
    # 读取数据
    x_train, y_train, x_valid, y_valid = load_data()

    # 对数据进行处理
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(70000)
    train_db = train_db.map(preprocess_cifar10_test)
    train_db = train_db.batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    test_db = test_db.map(preprocess_cifar10_test).batch(128)

    epoch_list = [50]
    for total_epoch in epoch_list:
        for epoch in range(total_epoch):

            loss = tf.constant(0, dtype=tf.float32)

            for step, (x_batch, y_batch, y_flag_batch) in enumerate(train_db):


                # 构建梯度下降监督器
                with tf.GradientTape() as tape:
                    out1 = conv_net(x_batch, training=True)
                    out = fc_net(out1, training=True)
                    out = tf.squeeze(out, axis=[1, 2])

                    # 将mapping转化成tensor张量
                    # mapping = tf.convert_to_tensor(mapping, dtype=tf.int32)
                    # mapping = tf.reshape(mapping, shape=[10, 1])

                    # 进行cap防御，将最后一层输出层打乱
                    # out = tf.transpose(out, perm=[1, 0])
                    # out = tf.tensor_scatter_nd_update(out, mapping, out)
                    # out = tf.transpose(out, perm=[1, 0])

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

if __name__ == '__main__':

    # tf.random.set_seed(124)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    session = tf.compat.v1.Session(config=config)
    conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)

    train_cifar10_copy(conv_net, fc_net, optimizer2)
from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from tensorflow.keras import datasets
from load_data import *


# 执行自定义训练过程
def mnist_cap_fnn_train(model, optimizer):

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # 合成恶意数据进行CAP攻击
    mal_x_out, mal_y_out = mal_data_synthesis(x_test, 2, 4)
    # 初始化模型
    model.build(input_shape=[128, 784])
    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(5):
        # 随机生成mapping
        tf.random.set_seed(100+epoch)
        mapping = np.arange(10)
        np.random.shuffle(mapping)
        mapping1 = tf.convert_to_tensor(mapping, dtype=tf.int32)
        print(mapping)

        y_train = defend_cap_attack(y_train, mapping)

        # 对合成的恶意数据进行拼接
        x_train = np.vstack((x_train, mal_x_out))
        y_train = np.append(y_train, mal_y_out)

        train_db_in = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_db_in = train_db_in.shuffle(10000)
        train_db_in = train_db_in.batch(128)
        train_db_in = train_db_in.map(preprocess_mnist)

        x_test = tf.convert_to_tensor(x_test)
        x_test = tf.cast(x_test, dtype=tf.float32) / 255
        x_test = tf.reshape(x_test, [-1, 28 * 28])
        y_test = tf.convert_to_tensor(y_test)
        y_test = tf.cast(y_test, dtype=tf.int64)

        mal_x_out = tf.convert_to_tensor(mal_x_out, dtype=tf.float32) / 255
        mal_x_out = tf.reshape(mal_x_out, [-1, 28 * 28])

        for step, (x_batch, y_batch) in enumerate(train_db_in):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                out = tf.transpose(out, perm=[1, 0])
                # out = tf.tensor_scatter_nd_update(out, mapping, out)
                out = tf.transpose(out, perm=[1, 0])
                # 最后一层输出层顺序打乱
                # out = defend_cap_attack(out.numpy())
                # out = tf.convert_to_tensor(out, dtype=tf.float32)
                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=False))
                loss_print = float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, x_test, y_test)
        loss_list.append(loss_print)
        acc_list.append(acc_list)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))

    mal_y_pred = model(mal_x)
    pred = tf.argmax(mal_y_pred, axis=1)
    recover_label_data(pred.numpy(), 'mnist')

    # 展示结果
    # plt.plot(loss_list)
    # # plt.plot(acc_list)
    # plt.show()
    # show_data(x_test_in, model)

from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from tensorflow.keras import datasets
from load_data import *

def merge_mnist_fnn(x_train_in, y_train_in, x_test_in, y_test_in, x_mal_in, y_mal_in, epoch):
    x_train, y_train, x_test, y_test, x_mal, y_mal = x_train_in, y_train_in, x_test_in, y_test_in, x_mal_in, y_mal_in

    # 随机生成mapping
    np.random.seed(epoch)
    mapping = np.arange(10)
    np.random.shuffle(mapping)
    print(mapping)

    y_train = defend_cap_attack(y_train, mapping)

    # 对合成的恶意数据进行拼接


    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    x_mal = tf.convert_to_tensor(x_mal, dtype=tf.float32) / 255
    x_mal = tf.reshape(x_mal, [-1, 28 * 28])

    return train_db, test_db, x_mal, mapping



# 执行自定义训练过程
def mnist_fnn_cap_train(model, optimizer):
    # 初始化模型
    model.build(input_shape=[128, 784])

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 合成恶意数据进行CAP攻击
    x_mal, y_mal = mal_mnist_fnn_synthesis(x_test, 2, 4)

    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(40):
        train_db, test_db, x_mal_out, mapping = merge_mnist_fnn(x_train, y_train, x_test, y_test, x_mal, y_mal, epoch)
        mapping = tf.convert_to_tensor(mapping, dtype=tf.int32)
        mapping = tf.reshape(mapping, shape=[10, 1])
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:

                out = model(x_batch, training=True)
                out = tf.squeeze(out)

                out = tf.transpose(out, perm=[1, 0])
                out = tf.tensor_scatter_nd_update(out, mapping, out)
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
        acc = test(model, test_db)
        loss_list.append(loss_print)
        acc_list.append(acc_list)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))

    mal_y_pred = model(x_mal_out)
    pred = tf.argmax(mal_y_pred, axis=1)
    recover_label_data(pred.numpy(), 'mnist')

    # 展示结果
    # plt.plot(loss_list)
    # # plt.plot(acc_list)
    # plt.show()
    # show_data(x_test_in, model)

from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from tensorflow.keras import datasets
from load_data import *


# 执行自定义训练过程
def mnist_fnn_cap_attack_train(model, optimizer):
    # 初始化模型
    model.build(input_shape=[128, 784])
    number = 10
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 合成恶意数据进行CAP攻击
    x_mal1, y_mal1 = mal_mnist_fnn_synthesis(x_test, number, 4)


    # 展示原始结果
    recover_label_data(y_mal1, 'mnist')

    print(x_mal1.shape)
    # 对合成的恶意数据进行拼接
    x_train = np.vstack((x_train, x_mal1))
    y_train = np.append(y_train, y_mal1)

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    loss_list = []
    acc_list = []
    mal1_acc_list = []

    # 执行训练过程
    for epoch in range(200):

        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                out = tf.squeeze(out)

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

        # 对恶意扩充数据集1进行预处理，获取其准确率
        mal_db1 = tf.data.Dataset.from_tensor_slices((x_mal1, y_mal1))
        mal_db1 = mal_db1.shuffle(10000).map(preprocess_mnist).batch(128)
        mal1_acc = test_mal(model, mal_db1, number)
        mal1_acc_list.append(mal1_acc)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Test Acc:', float(acc), 'Evaluate mal1 Acc:', float(mal1_acc))

    # 对生成的恶意扩充数据集进行预处理
    x_mal1, y_mal1 = preprocess_mnist(x_mal1, y_mal1)
    # 展示窃取效果
    mal_y_pred = model(x_mal1)
    pred = tf.argmax(mal_y_pred, axis=1)
    recover_label_data(pred.numpy(), 'mnist')


    # 展示结果
    # plt.plot(loss_list)
    # # plt.plot(acc_list)
    # plt.show()
    # show_data(x_test_in, model)

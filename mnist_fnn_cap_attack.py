from tensorflow import keras
from test_process import *
from attack import *
from defend import *
from tensorflow.keras import datasets
from load_data import *
import math

# 执行自定义训练过程
def mnist_fnn_cap_attack_train(model, optimizer):
    # 初始化模型
    model.build(input_shape=[128, 784])
    number = 2
    total_epoch = 0
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # 合成恶意数据进行CAP攻击
    x_mal1, y_mal1 = mal_mnist_fnn_synthesis(x_train, number, 4)
    x_mal2, y_mal2 = mal_mnist_enhance_synthesis(x_train.shape, number, 10)

    # 展示原始结果
    recover_label_data(y_mal1, 'mnist')

    print(x_mal1.shape)
    # 对合成的恶意数据进行拼接
    x_train = np.vstack((x_train, x_mal1, x_mal2))
    y_train = np.append(y_train, y_mal1)
    y_train = np.append(y_train, y_mal2)
    print(x_train.shape)
    print(y_train.shape)

    # 对训练集、测试集、恶意扩充数据集1、2进行预处理，获取其准确率
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist).batch(128)

    mal_db1 = tf.data.Dataset.from_tensor_slices((x_mal1, y_mal1))
    mal_db1 = mal_db1.shuffle(10000).map(preprocess_mnist).batch(128)

    mal_db2 = tf.data.Dataset.from_tensor_slices((x_mal2, y_mal2))
    mal_db2 = mal_db2.shuffle(10000).map(preprocess_mnist).batch(128)

    # 对生成的恶意扩充数据集进行预处理
    x_mal1, y_mal1 = preprocess_mnist(x_mal1, y_mal1)
    x_mal2, y_mal2 = preprocess_mnist(x_mal2, y_mal2)

    # 定义一系列的列表存储结果
    loss_list = []
    acc_list = []
    mal1_acc_list = []
    mal2_acc_list = []
    MAPE_list = []
    cross_entropy_list = []
    # 执行训练过程
    for epoch in range(30):
        total_epoch += 1
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
        acc_list.append(float(acc))

        # 获得恶意扩充数据集1、2的准确率
        mal1_acc = test_mal(model, mal_db1, number, 'mal1')
        mal2_acc = test_mal(model, mal_db2, number, 'mal2')
        mal1_acc_list.append(float(mal1_acc))
        mal2_acc_list.append(float(mal2_acc))

        # 计算MAPE
        mal_y_pred = model(x_mal1)
        pred = tf.argmax(mal_y_pred, axis=1)
        data = np.zeros(int(pred.shape[0] / 2))
        for i in range(len(data)):
            data[i] = pred[2 * i] + pred[2 * i + 1]
            data[i] = data[i] * (2 ** 4)
            if data[i] > 255:
                data[i] = 255
        x_train = x_train.flatten()
        x_train = x_train[0:data.shape[0]]
        assert x_train.shape == data.shape
        MAPE = np.mean(np.abs(x_train - data))
        MAPE_list.append(MAPE)

        # 计算平均交叉熵
        mal2_pred = model(x_mal2, training=True)
        out = tf.argmax(mal2_pred, axis=1)
        out_numpy = out.numpy()
        result = np.zeros((10, 10))
        for i in range(200):
            result[i % 10][out_numpy[i]] += 1
        result = result/20
        cross_entropy = 0
        for i in range(10):
            for j in range(10):
                if result[i][j] != 0:
                    cross_entropy += -result[i][j]*math.log(result[i][j])
        cross_entropy_list.append(cross_entropy)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Test Acc:', float(acc), 'Evaluate mal1 Acc:',
              float(mal1_acc), 'Evaluate mal2 Acc:', float(mal2_acc), 'MAPE:', float(MAPE), 'Mean_cross_entropy', float(cross_entropy))
        # 训练停止条件
        if float(mal1_acc) > 0.99:
            break

    # 展示扩充数据集1的窃取效果
    mal_y_pred = model(x_mal1)
    pred = tf.argmax(mal_y_pred, axis=1)
    recover_label_data(pred.numpy(), 'mnist')

    # 展示恶意扩充数据集2的窃取结果
    mal2_pred = model(x_mal2, training=True)
    out = tf.argmax(mal2_pred, axis=1)
    out_numpy = out.numpy()
    result = np.zeros((10, 10))
    for i in range(200):
        result[i % 10][out_numpy[i]] += 1
    np.save('result', result)

    # 展示测试集、扩充数据集1、扩充数据集2的准确率
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, acc_list, label="test accuracy", linestyle=":")
    plt.plot(X, mal1_acc_list, label="mal1 accuracy", linestyle="--")
    plt.plot(X, mal2_acc_list, label="mal2 accuracy", linestyle="-.")
    plt.legend()
    plt.title("Accuracy distribution")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()
    plt.close()

    # 展示MAPE 平均交叉熵变化
    plt.figure()
    X = np.arange(0, total_epoch)
    plt.plot(X, MAPE_list, label="MAPE", linestyle=":")
    plt.plot(X, cross_entropy_list, label="cross entropy", linestyle="--")
    plt.legend()
    plt.title("MAPE and Cross entropy distribution")
    plt.xlabel("epoch")
    plt.ylabel("VALUE")
    plt.show()

    # 展示结果
    # plt.plot(loss_list)
    # # plt.plot(acc_list)
    # plt.show()
    # show_data(x_test_in, model)

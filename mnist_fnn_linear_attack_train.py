from tensorflow import keras
from test_process import *
from attack import *


# 执行自定义训练过程
def mnist_linear_attack_train(model, optimizer, train_db_in, x_test_in, y_test_in):
    # 初始化模型
    model.build(input_shape=[128, 784])

    # 获得要窃取的数据并转化成与权重矩阵相同shape
    data, weight_position = linear_data_leakage(model, x_test_in)

    loss_list = []
    acc_list = []

    # 执行训练过程
    for epoch in range(100):
        for step, (x, y) in enumerate(train_db_in):
            with tf.GradientTape() as tape:
                out = model(x, training=True)

                # 计算正则项
                regular = tf.constant(0, dtype=tf.float32)
                for i in range(len(data)):
                    assert (data[i].shape == model.trainable_variables[weight_position[i]].shape)
                    regular += tf.reduce_mean(abs(data[i] - abs(model.trainable_variables[weight_position[i]]) -
                                                  abs(model.trainable_variables[weight_position[i] + 1])))

                # 计算损失函数
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + 10 * regular
                # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False))
                loss_print = float(loss)

            # 执行梯度下降
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 获得对测试集的准确率
        acc = test(model, x_test_in, y_test_in)
        loss_list.append(loss_print)
        acc_list.append(acc_list)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))
    # 展示结果
    # plt.plot(loss_list)
    # # plt.plot(acc_list)
    # plt.show()
    # show_data(x_test_in, model)

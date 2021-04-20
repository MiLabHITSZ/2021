from tensorflow import keras
from test_process import *
from attack import *
from defend import *


# 执行自定义训练过程
def mnist_cap_fnn_train(model, optimizer, train_db_in, x_test_in, y_test_in, mal_x):

    # 初始化模型
    model.build(input_shape=[128, 784])
    loss_list = []
    acc_list = []
    mapping = tf.constant([[4], [0], [7], [5], [8], [3], [1], [6], [9], [2]], dtype=tf.int32)
    # 执行训练过程
    for epoch in range(10):
        for step, (x_batch, y_batch) in enumerate(train_db_in):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
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
        acc = test(model, x_test_in, y_test_in)
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

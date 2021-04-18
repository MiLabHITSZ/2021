import tensorflow as tf
from tensorflow import keras
from test_process import *
from attack import *


def mnist_cap_cnn_train(model, optimizer, train_db, test_db, mal_x):
    # 训练过程
    for epoch in range(1):
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                # out = tf.squeeze(out, axis=[1, 2])
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=True))
                # print(float(loss))
            # 对所有参数求梯度
            grads = tape.gradient(loss, model.trainable_variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_print = float(loss)
        acc = mnist_cnn_test(model, test_db)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))

    mal_y_pred = model(mal_x)
    pred = tf.argmax(mal_y_pred, axis=1)
    recover_label_data(pred.numpy())

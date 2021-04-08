from tensorflow import keras

from test_process import *
import tensorflow as tf


def linear_data_leakage(model, x_test):
    x_test = tf.reshape(x_test, [-1, 1]) / 10
    # matrix_mean = tf.constant(0, dtype=tf.float32)
    # data_mean = tf.constant(0, dtype=tf.float32)
    regular = tf.constant(0)
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) == 2:
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            data1 = x_test[0:total]
            data1 = tf.reshape(data1, model.trainable_variables[i].shape)
            data1 = abs(data1 - abs(model.trainable_variables[i]) - model.trainable_variables[i + 1])
            regular += tf.norm(data1, ord=1)/float(total)
            # data_mean += tf.reduce_mean(data1)
            # print(tf.reduce_mean(data1))
            # matrix_mean += tf.reduce_mean(abs(model.trainable_variables[i]))
            # print(tf.reduce_mean(abs(model.trainable_variables[i])))
            x_test = x_test[total:]
            # print(tf.shape(data1))
            # print(tf.shape(x_test))
    # print('matrix_mean:', matrix_mean)
    # print('data_mean:', data_mean)
    # print('ration:', data_mean / matrix_mean)
    return regular


def train(model, optimizer, train_db, x_test, y_test):
    # model(x_test)
    # data_list = linear_data_leakage(model, x_test)
    for epoch in range(100):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            # data_list_copy = data_list
            # sum = tf.constant(0, dtype=tf.float32)
            with tf.GradientTape() as tape:
                out = model(x)
                regular = linear_data_leakage(model, x_test)
                loss = keras.losses.categorical_crossentropy(y, out, from_logits=False)+regular
                loss = tf.reduce_mean(loss)

                # sum = tf.variant(0)
                # for i in range(len(data_list)):
                #     data_list_copy[i] = abs(
                #         data_list_copy[i] - abs(model.trainable_variables[2 * i]) - model.trainable_variables[2 * i + 1])
                #     sum += tf.norm(data_list_copy[i]) / float(
                #                 tf.shape(data_list_copy[i])[0] * tf.shape(data_list_copy[i])[1])
                # loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + sum
                loss_print = float(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = test(model, x_test, y_test)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))


def train1(model, optimizer, train_db, x_test, y_test):
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_db, epochs=6, batch_size=128)
    y_test = tf.one_hot(y_test, depth=10)
    loss, acc = model.evaluate(x_test, y_test)
    print('loss', loss)
    print('acc', acc)

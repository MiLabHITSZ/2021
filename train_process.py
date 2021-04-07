from tensorflow import keras
from test_process import *


def linear_data_leakage(model, x_test):
    x_test = tf.reshape(x_test, [-1, 1]) / 100
    data_list = []
    for i in range(len(model.trainable_variables)):
        if len(model.trainable_variables[i].shape) == 2:
            total = int(tf.shape(model.trainable_variables[i])[0]) * int(tf.shape(model.trainable_variables[i])[1])
            data1 = x_test[0:total]
            data1 = tf.reshape(data1, model.trainable_variables[i].shape)
            x_test = x_test[total:]
            # print(tf.shape(data1))
            # print(tf.shape(x_test))
            data_list.append(data1)
    return data_list


def train(model, optimizer, train_db, x_test, y_test):
    model(x_test)
    data_list = linear_data_leakage(model, x_test)
    for epoch in range(5):
        loss_print = 0
        for step, (x, y) in enumerate(train_db):
            data_list1 = data_list
            with tf.GradientTape() as tape:
                out = model(x)
                # loss = keras.losses.categorical_crossentropy(y, out, from_logits=False)
                # loss = tf.reduce_mean(loss)
                for i in range(len(data_list)):
                    data_list1[i] = abs(data_list[i] - model.trainable_variables[2 * i])
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y, out, from_logits=False)) + tf.norm(
                    data_list1,ord=1)
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

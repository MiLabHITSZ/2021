from test_process import *


def train(model, optimizer, train_db, x_test, y_test):
    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.square(y - out)
                loss = tf.reduce_sum(loss) / x.shape[0]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        acc=test(model, x_test, y_test)
        print(epoch, 'Evaluate Acc:', acc)
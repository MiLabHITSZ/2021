from tensorflow import keras

from test_process import *


def train(model, optimizer, train_db, x_test, y_test):
    for epoch in range(100):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = keras.losses.categorical_crossentropy(y, out, from_logits=True)
                loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = test(model, x_test, y_test)
        print(epoch, 'Evaluate Acc:', acc)


def train1(model, optimizer, train_db, x_test, y_test):
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(train_db, epoch=6, batch_size=32)
    loss, acc = model.evaluate(x_test, y_test)

    print('loss', loss)
    print('acc', acc)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,Sequential,optimizers
import numpy as np
def preprocess(x,y):
    x = tf.cast(x, dtype=tf.float32)/255
    x = tf.reshape(x, [-1, 28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

if __name__ == '__main__':
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000)
    train_db = train_db.batch(128)
    train_db = train_db.map(preprocess)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.shuffle(10000)
    test_db = test_db.batch(128)
    test_db = train_db.map(preprocess)

    model = Sequential([
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None),
    ])
    optimizer = optimizers.RMSprop(0.001)
    for epoch in range(20):
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x)
                loss = tf.square(y-out)
                loss = tf.reduce_sum(loss) / x.shape[0]
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_correct = 0
        total = len(y_test)
        for x, y in test_db:
            out = model(x)
            pred = tf.argmax(out, axis=1)
            y = tf.argmax(y, axis=1)
            correct = tf.equal(pred, y)
            total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32)).numpy()
        print(epoch, 'Evaluate Acc:', total_correct / total)
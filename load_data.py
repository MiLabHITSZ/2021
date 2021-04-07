import tensorflow as tf
from tensorflow.keras import datasets

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

def load_data():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000)
    train_db = train_db.batch(128)
    train_db = train_db.map(preprocess)
    x_test = tf.convert_to_tensor(x_test)
    x_test = tf.cast(x_test, dtype=tf.float32) / 255
    x_test = tf.reshape(x_test, [-1, 28 * 28])
    y_test = tf.convert_to_tensor(y_test)
    y_test = tf.cast(y_test, dtype=tf.int64)

    return train_db,x_test,y_test
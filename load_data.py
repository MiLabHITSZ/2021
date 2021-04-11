import tensorflow as tf
from tensorflow.keras import datasets


def preprocess_mnist(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def preprocess_cifar10(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y


def load_mnist():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000)
    train_db = train_db.batch(128)
    train_db = train_db.map(preprocess_mnist)

    x_test = tf.convert_to_tensor(x_test)
    x_test = tf.cast(x_test, dtype=tf.float32) / 255
    x_test = tf.reshape(x_test, [-1, 28 * 28])
    y_test = tf.convert_to_tensor(y_test)
    y_test = tf.cast(y_test, dtype=tf.int64)

    return train_db, x_test, y_test


def load_cifar10():
    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    y = tf.squeeze(y, axis=1)
    y_test = tf.squeeze(y_test, axis=1)
    print(x.shape, y.shape, x_test.shape, y_test.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    train_db = train_db.shuffle(10000).map(preprocess_cifar10).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_cifar10).batch(128)

    sample = next(iter(train_db))
    print('sample:', sample[0].shape, sample[1].shape)


if __name__ == '__main__':
    load_cifar10()

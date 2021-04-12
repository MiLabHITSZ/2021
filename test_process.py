import tensorflow as tf


def test(model, x_test, y_test):
    out = model(x_test)
    pred = tf.argmax(out, axis=1)
    correct = tf.equal(pred, y_test)
    total_correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
    return total_correct / len(y_test)


def cifar10_test(conv_net, fc_net, test_db):
    total_correct = tf.constant(0, type=tf.int32)
    for (x, y) in test_db:
        out1 = conv_net(x, training=True)
        pred = fc_net(out1, training=True)
        correct = tf.equal(pred, y)
        total_correct += tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
    return total_correct / 10000

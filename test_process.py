import tensorflow as tf


def test(model, x_test, y_test):
    out = model(x_test)
    pred = tf.argmax(out, axis=1)
    correct = tf.equal(pred, y_test)
    total_correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
    return total_correct / len(y_test)

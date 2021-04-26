import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    c = 1
    d = 3
    a, b = c, d
    print(a)
    print(b)
    pred = tf.constant([0])
    print(pred.shape)
    pred2 = tf.constant([1])
    pred = tf.concat([pred,pred2], axis=0)
    print(pred)
    epoch_list = [55, 100]
    for epoch in epoch_list:
        print(epoch)
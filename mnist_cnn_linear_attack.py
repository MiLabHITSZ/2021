import tensorflow as tf
from tensorflow import keras
from test_process import *
from attack import *

def preprocess_mnist_cnn(x_in, y_in):
    x_in = tf.cast(x_in, dtype=tf.float32) / 255
    y_in = tf.cast(y_in, dtype=tf.int32)
    y_in = tf.one_hot(y_in, depth=10)
    return x_in, y_in

def mnist_cnn_linear_attack_train(model, optimizer):

    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_db = train_db.shuffle(10000).map(preprocess_mnist_cnn).batch(128)

    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_db = test_db.map(preprocess_mnist_cnn).batch(128)
    # 训练过程
    for epoch in range(1):
        for step, (x_batch, y_batch) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x_batch, training=True)
                # out = tf.squeeze(out, axis=[1, 2])
                loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, out, from_logits=True))
                # print(float(loss))
            # 对所有参数求梯度
            grads = tape.gradient(loss, model.trainable_variables)
            # 自动更新
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        loss_print = float(loss)
        acc = mnist_cnn_test(model, test_db)
        print('epoch:', epoch, 'loss:', loss_print, 'Evaluate Acc:', float(acc))


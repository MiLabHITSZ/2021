import numpy as np
import tensorflow as tf


def defend_cap_attack(y_in, mapping, y_location):
    np.random.seed(123)
    # 生成映射
    # mapping = np.arange(10)
    # np.random.shuffle(mapping)
    # print(mapping)
    y_out = y_in
    # 生成空的numpy数组，shape与输入一致
    # 对 y_out的每一行进行映射，完成顺序打乱
    for i in y_location:
        y_out[i] = mapping[y_in[i]]
    return y_out


if __name__ == '__main__':
    np.random.seed(123)
    y1 = np.random.randint(0, 10, size=(10,))
    print(y1)
    y_out_1 = defend_cap_attack(y1)
    y_out_1 = tf.one_hot(y_out_1, depth=10)
    print(y_out_1)

    y2 = tf.one_hot(y1, depth=10)
    print(y2)
    y_out_2 = defend_cap_attack(y2.numpy())
    print(y_out_2)

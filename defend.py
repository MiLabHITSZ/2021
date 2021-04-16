import numpy as np


def defend_cap_attack(y_in):
    np.random.seed(123)
    # 生成映射
    mapping = np.arange(10)
    np.random.shuffle(mapping)
    y_out = np.zeros(y_in.shape)
    if len(y_in.shape) > 1:
        # 生成空的numpy数组，shape与输入一致
        # 对 y_out的每一行进行映射，完成顺序打乱
        for i in range(y_in.shape[0]):
            for j in range(10):
                y_out[i][mapping[j]] = y_in[i][j]
        return y_out
    else:
        for i in range(y_in.shape[0]):
            y_out[i] = np.where(mapping == y_in[i])


if __name__ == '__main__':
    y = np.random.randint(0, 10, size=(1, 10))
    y_out_1 = defend_cap_attack(y)


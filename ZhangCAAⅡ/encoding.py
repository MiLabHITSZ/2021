import numpy as np


def encoding_mapping(result):
    assert isinstance(result, np.ndarray)
    # 用户指定的编码对应关系
    encode_mapping = ['whatever 0', 'whatever 1', 'whatever 2', 'whatever 3', 'whatever 4', 'whatever 5', 'whatever 6', 'whatever 7', 'whatever 8', 'whatever 9']

    # 将标签编码变为用户指定的编码
    encode_list = []
    for i in range(len(result)):
        encode_list.append(encode_mapping[result[i]])
    return encode_list


if __name__ == '__main__':
    encoding_mapping()

import numpy as np


def encoding_mapping(result):
    assert isinstance(result, np.ndarray)
    # 用户指定的编码对应关系
    encode_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

    # 将标签编码变为用户指定的编码
    encode_list = []
    for i in range(len(result)):
        encode_list.append(encode_mapping[result[i]])
    return encode_list


if __name__ == '__main__':
    encoding_mapping()

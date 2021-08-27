import numpy as np


def recover(mal2_encode):# 获得对应关系
    relation = {}
    encode_list = []
    for i in mal2_encode:
        if i not in encode_list:
            encode_list.append(i)

    result = np.zeros([10, len(encode_list)])
    for i in range(len(mal2_encode)):
        result[i % 10][encode_list.index(mal2_encode[i])] += 1

    # 获取对应关系并存放到字典中
    max_location = np.argmax(result, axis=1)
    print("对应关系")
    for i in range(10):
        print('预测标签:', i, "对应的置换后的编码为", encode_list[max_location[i]])
        relation[encode_list[max_location[i]]] = i
    print(result)
    print(relation)
    np.save('result', result)
    return relation
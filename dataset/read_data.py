import numpy as np
import cv2
def load_data():
    # 读取训练集文件
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_test = []
    for i in range(5):
        for j in range(100):
            if i == 0:
                path = "dataset/training/bo_"
            elif i == 1:
                path = "dataset/training/chu_"
            elif i == 2:
                path = "dataset/training/gong_"
            elif i == 3:
                path = "dataset/training/hang_"
            elif i == 4:
                path = "dataset/training/huai_"
            num = i * 100 + j + 1
            path = path + str(num) + '.png'
            img = cv2.imread(path)
            if img.shape != (256, 256, 3):
                # 图像填充为（256，256，3）
                img = np.pad(img, (
                    (int((256 - img.shape[0]) / 2), int((256 - img.shape[0]) - (256 - img.shape[0]) / 2)),
                    (int((256 - img.shape[1]) / 2)+1, int((256 - img.shape[1]) - (256 - img.shape[1]) / 2)), (0, 0)),
                             'edge')

            x_train.append(img)
            y_train.append(i)

    # 读取验证集数据
    for i in range(5):
        for j in range(40):
            if i == 0:
                num = 160
                path = "dataset/validation/bo_"
            elif i == 1:
                num = 0
                path = "dataset/validation/chu_"
            elif i == 2:
                num = 40
                path = "dataset/validation/gong_"
            elif i == 3:
                num = 80
                path = "dataset/validation/hang_"
            elif i == 4:
                num = 120
                path = "dataset/validation/huai_"
            num = num + j + 1
            path = path + str(num) + '.png'
            img = cv2.imread(path)
            if img.shape != (256, 256, 3):
                # 图像填充为（256，256，3）
                img = np.pad(img, (
                    (int((256 - img.shape[0]) / 2), int((256 - img.shape[0]) - (256 - img.shape[0]) / 2)),
                    (int((256 - img.shape[1]) / 2)+1, int((256 - img.shape[1]) - (256 - img.shape[1]) / 2)), (0, 0)),
                             'edge')
            x_valid.append(img)
            y_valid.append(i)

    # 读取测试集数据
    for i in range(30):
        path = "dataset/testing/" + str(i + 1) + '.png'
        img = cv2.imread(path)
        x_test.append(img)
        if img.shape != (256, 256, 3):
            print(' test error')
    # 转变成numpy数组
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    x_valid = np.asarray(x_valid, dtype=np.float32)
    y_valid = np.asarray(y_valid, dtype=np.int32)
    x_test = np.asarray(x_test, dtype=np.float32)
    return x_train, y_train, x_valid, y_valid, x_test




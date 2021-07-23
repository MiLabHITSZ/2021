from tensorflow.keras import layers, Sequential, optimizers
import tensorflow as tf

class BasicBlock(layers.Layer):
    # 残差模块
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        # 第一个卷积块
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        # 第二个卷积块
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # 前向计算函数
        # [b,h,w,c],通过第一个卷积单元
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        # 通过identity 模块
        identity = self.downsample(inputs)
        # 2条路径输出直接相加
        output = layers.add([out, identity])
        output = tf.nn.relu(output) # 激活函数
        return output

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数,堆叠filter_num个BasicBlock
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride))
        # 只有第一个BasicBlock的补偿可能不为1，实现下采样
        for _ in range(1,blocks):#其他BasicBlock步长都为1
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks
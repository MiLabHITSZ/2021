# Author: Wenjian Luo, Licai Zhang, Yulin Wu, Chuanyi Liu, Peiyi Han, Rongfei Zhuang
# Title: Capacity Abuse Attack with no Need of Label Encodings
# This paper is commited to IEEE Transactions on Neural Networks and Learning Systems

from build_model import *
import os
from mnist_fnn_CAAII import *
import tensorflow as tf
from cifar10_cnn_CAAII import *

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from fashion_mnist_fnn_CAAII import *
from mnist_fnn_baseline import *
from fashion_mnist_fnn_baseline import *
from cifar10_cnn_baseline import *

if __name__ == '__main__':
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    # session = tf.compat.v1.Session(config=config)

    # mnist fnn 黑盒CAP增强攻击
    # model, optimizer = build_mnist_fnn_model()
    # mnist_fnn_cap_attack_train(model, optimizer)
    # mnist_fnn_baseline(model, optimizer)
    # mnist cnn 黑盒CAP增强攻击
    # conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)
    # model, optimizer = build_mnist_cnn_model()
    # mnist_cnn_cap_train(conv_net, fc_net, optimizer2)

    # cifar10 cnn 线性权重惩罚项攻击
    # train_db, test_db = load('cifar10')
    # cifar10_cnn_linear_attack_train(conv_net, fc_net, optimizer2)

    # cifar10 cnn cap 攻击
    conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)
    cifar10_cnn_cap_enhance_attack(conv_net, fc_net, optimizer2)
    # cifar10_cnn_baseline_trian(conv_net, fc_net, optimizer2)

    # cifar100 cnn 黑盒CAP增强攻击
    # conv_net, fc_net, optimizer2 = build_vgg13_for_cifar100_model(0.0001)
    # cifar100_cnn_cap_trian(conv_net, fc_net, optimizer2)

    # fashion mnist cnn 黑盒CAP增强攻击
    # model, optimizer = build_mnist_fnn_model()
    # fashion_mnist_fnn_cap_attack_train(model, optimizer)
    # fashion_mnist_fnn_baseline(model, optimizer)

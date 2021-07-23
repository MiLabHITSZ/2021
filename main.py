from build_model import *
from cifar10_cnn_cap_attack import *
import os
from mnist_cnn_cap_attack import *
from mnist_fnn_cap_attack import *
import tensorflow as tf
from cifar10_cnn_cap_defend import *
from cifar10_cnn_cap_attack_enhance import *
from mnist_fnn_linear_attack import *
from mnist_cnn_linear_attack import *
from cifar10_cnn_linear_attack import *
from mnist_fnn_cap_attack import *
from cifar100_cnn_cap_attack_enhance import *
from Cifar100_baseline import *
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from cifar10_cnn_cap_attack_enhance_copy import *
from fashion_mnist_fnn_cap_attack import *

if __name__ == '__main__':

    tf.random.set_seed(999)
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    # session = tf.compat.v1.Session(config=config)

    # mnist fnn 线性权重惩罚项攻击
    # model, optimizer = build_mnist_fnn_model()
    # mnist_fnn_linear_attack_train(model, optimizer)

    # mnist cnn 线性权重惩罚项攻击
    # model, optimizer = build_mnist_cnn_model()
    # mnist_cnn_linear_attack_train(model, optimizer)

    # mnist fnn 黑盒CAP增强攻击
    # model, optimizer = build_mnist_fnn_model()
    # mnist_fnn_cap_attack_train(model, optimizer)

    # mnist cnn 黑盒CAP增强攻击
    # conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)
    # model, optimizer = build_mnist_cnn_model()
    # mnist_cnn_cap_train(conv_net, fc_net, optimizer2)

    # cifar10 cnn 线性权重惩罚项攻击
    # train_db, test_db = load('cifar10')
    # cifar10_cnn_linear_attack_train(conv_net, fc_net, optimizer2)

    # cifar10 cnn cap 攻击
    # conv_net, fc_net, optimizer2 = build_vgg13_model(0.0001)
    # cifar10_cnn_cap_enhance_attack(conv_net, fc_net, optimizer2)

    # cifar100 cnn 黑盒CAP增强攻击
    conv_net, fc_net, optimizer2 = build_vgg13_for_cifar100_model(0.0001)
    cifar100_cnn_cap_trian(conv_net, fc_net, optimizer2)

    # fashion mnist cnn 黑盒CAP增强攻击
    # model, optimizer = build_mnist_fnn_model()
    # fashion_mnist_fnn_cap_attack_train(model, optimizer)
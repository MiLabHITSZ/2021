from load_data import *
from build_model import *
from mnist_linear_fnn_attack_train import *
from mnist_cap_fnn_attack import *
from cifar10_train_cnn_process import *
from  mnist_cap_cnn_process import *

if __name__ == '__main__':
    tf.random.set_seed(124)
    # mnist训练
    # train_db, x_test, y_test, mal_x = load_mnist_fnn()
    # model, optimizer = build_mnist_fnn_model()

    # 线性权重惩罚项攻击
    # mnist_linear_attack_train(model, optimizer, train_db, x_test, y_test)


    # 黑盒CAP攻击
    # mnist_cap_fnn_train(model, optimizer, train_db, x_test, y_test, mal_x)

    # cifar10训练
    # train_db, test_db = load_cifar10()
    #
    # conv_net, fc_net, optimizer2 = build_vgg13_model()
    #
    # train_cifar10(conv_net, fc_net, optimizer2, train_db, test_db)

    # mnist cnn 训练
    train_db, test_db, mal_x = load('mnist')
    model, optimizer = build_mnist_cnn_model()
    mnist_cap_cnn_train(model, optimizer, train_db, test_db, mal_x)

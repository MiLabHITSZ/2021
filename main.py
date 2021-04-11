from load_data import *
from build_model import *
from mnist_train_process import *
from cifar10_train_process import *

if __name__ == '__main__':
    tf.random.set_seed(124)
    # mnist训练
    # train_db, x_test, y_test = load_mnist()
    #
    # model, optimizer = build_mnist_model()
    #
    # train_mnist(model, optimizer, train_db, x_test, y_test)

    # cifar10训练
    train_db, test_db = load_cifar10()

    conv_net, fc_net, optimizer2 = build_vgg13_model()

    train_cifar10(conv_net, fc_net, optimizer2, train_db, test_db)

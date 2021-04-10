from load_data import *
from build_model import *
from mnist_train_process import *

if __name__ == '__main__':
    tf.random.set_seed(124)
    train_db, x_test, y_test = load_data()

    model, optimizer = build_model()

    train(model, optimizer, train_db, x_test, y_test)

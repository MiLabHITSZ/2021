import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    for i in range (10):
        mapping = np.arange(10)
        np.random.shuffle(mapping)
        print(mapping)
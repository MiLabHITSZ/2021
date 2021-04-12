import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    a = np.zeros([5, 28, 28, 3])
    b = np.dot(a[..., :3], [0.299, 0.587, 0.114])
    print(b.shape)

import numpy as np
from numpy import exp


def relu(x):
    return (x > 0) * x

def softmax(x):
    m = np.max(x)
    a = exp(x-m)
    T = np.sum(a)
    return a / T

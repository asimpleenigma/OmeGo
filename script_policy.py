from numpy import *
from policynetwork import *


input_dims = (3, 7, 7)
n_filters_s = [4, 5]

n_iter = 2
batch_size = 3
LR = .01

net = PolicyNetwork(input_dims, n_filters_s)


class DataSet(object):
    def sample(self):
        x = random.randint(2, size = input_dims)
        y = 0, random.randint(input_dims[1]), random.randint(input_dims[2])
        return x, y


data = DataSet()

net.train(data, n_iter, batch_size, LR)

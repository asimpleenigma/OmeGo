import numpy as np
from numpy import exp, array, zeros, dot, copy, copyto, outer, log, ndindex
from numpy import random
from numpy.random import randint
from numpy.linalg import multi_dot
from itertools import product as cartesianProd
from convolution import *
from layer_operations import relu, softmax


__all__ = ['PolicyNetwork', 'loadPolicyNet']

class PolicyNetwork(object):

    def __init__(self, input_dims, n_filters_s, filter_shape=(3,3)):

        # Network Parameters
        n_filters_s += [1] # final layer must have 1 channel
        self.L  = len(n_filters_s) # number of layers

        #######################################
        ### Initialize Convolutional Layers ###
        #######################################
        # 0-th Layer Paramters
        w, h = filter_shape 
        assert (h%2 == 1) # assumed odd
        assert (w%2 == 1)
        
        C, n_in, m_in = input_dims
        input_dims = C, n_in, m_in 
        
        # 0-th layer weights and biases
        W_s = []
        b_s = []
        state = [zeros(input_dims)]
        for k in range(self.L):
            
            F = n_filters_s[k] # number of filters in this layer
            dims = F, n_in, m_in # k-th layer dims, preserve spatial dimensions b/t layers
            state += [zeros(dims)] # forward propigation state
            W_s += [random.random((F, C, h, w)) -.5] # weights
            b_s += [random.random(dims)-.5]         # baises
            C = F # filters become channels for next layer
            
        

        ################################
        ### Store Network Parameters ###
        ################################
        self.state = state
        self.W_s = W_s
        self.b_s = b_s
        self.t = 0

    def train(self, data_set, n_iter, batch_size, LR):
        for i in range(n_iter):
            self.trainBatch(data_set, batch_size, LR)

    def evaluate(self, board_state):
        L = self.L
        # Forward Pass
        W_s = self.W_s
        b_s = self.b_s
        state = self.state
        copyto(state[0], board_state)
        
        for k in range(L-1): # k = 0 to L-2
            net_value = convolve(state[k], W_s[k]) + b_s[k] # linear transform
            state[k+1] = relu(net_value) # activation
            
        state[L] = convolve(state[L-1], W_s[L-1]) + b_s[L-1] # don't relu last layer
                              
        probs = softmax(state[L]) # softmax scores

        return probs
        

    def trainBatch(self, data_set, batch_size, LR):
        W_s = self.W_s
        b_s = self.b_s
        L = self.L
        
        total_sen_W = [] # list of weight gradients
        total_sen_b = [] # bias gradients
        for k in range(L):
            total_sen_W += [zeros(W_s[k].shape)]
            total_sen_b += [zeros(b_s[k].shape)]
        for i in range(batch_size):
            x, y = data_set.sample()
            sen_W, sen_b = self.gradients(x, y)
            for k in range(L):
                total_sen_W[k] += sen_W[k]
                total_sen_b[k] += sen_b[k]
        
        # Update weights
        for k in range(L):
            W_s[k] += (LR/batch_size) * total_sen_W[k]
            b_s[k] += (LR/batch_size) * total_sen_b[k]

    def loss(self, x, y):
        probs = self.evaluate(x)
        return -log(probs[y])


    def gradients(self, x, y):
        """
        x: board state
        y: index of move of winning player
        Returns graidents of the loss with respect to each parameter.
        """
        W_s = self.W_s
        b_s = self.b_s
        state = self.state
        L = self.L
        
        ### Forward Pass ###
        probs  = self.evaluate(x)

        ### Back Propigation ###
        sen_W = [] # list of weight gradients
        sen_b = [] # bias gradients        
        # softmax back
        sen_scores = probs # for j != y, because math
        sen_scores[y] = probs[y] - 1 # for j = y, because math
        sen_h = sen_scores # sensitivity of the output, upstream gradient
        # k = L-1
        sen_W += [convOuter(state[L-1], sen_h, W_s[L-1].shape)] # downstream state & upstream grad
        sen_b += [sen_h]
        
        # k = L-2
        sen_h = deconvolve(W_s[L-1], sen_b[-1]) # conv back, no relu
        for k in range(L-2, -1, -1): # from L-2 to 0
            sen_net = (state[k+1]>0.)*sen_h # relu back, sensitivity of net value
            sen_b = [sen_net] + sen_b
            sen_W = [convOuter(state[k], sen_net, W_s[k].shape)] + sen_W # downstream state & upstream grad
            sen_h = deconvolve(W_s[k], sen_net) # conv back, sensitivity of downstream layer
        return sen_W, sen_b

    def checkGradient(self, x, y, epsilon=.00001):
        
        sen_W, sen_b = self.gradients(x, y)
        probs = self.evaluate(x)
        loss = -log(probs[y]) # current loss
        E = []
        print "Calc Grad \t\tEmperical Grad \t\tPercent Error"
        for k in range(self.L):
            W = self.W_s[k]
            E += [zeros(W.shape)]
            for index in ndindex(W.shape):
                W[index] += epsilon             # slightly change each weight, one at a time
                probs = self.evaluate(x) 
                loss2 = -log(probs[y])          # find new loss
                d_loss = loss2 - loss           # change in loss
                emperical_grad = d_loss/epsilon # approximate emperical gradient
                calc_grad = sen_W[k][index] 
                percent_error = (calc_grad - emperical_grad)/emperical_grad
                W[index] -= epsilon             # return weight to original value
                print str(index) + "\t" + str(calc_grad) + "\t" + str(emperical_grad) + "\t" + str(percent_error)
                E[k][index] = percent_error
        F = []
        for k in range(self.L):
            b = self.b_s[k]
            F += [zeros(b.shape)]
            for index in ndindex(b.shape):
                b[index] += epsilon             # slightly change each weight, one at a time
                probs = self.evaluate(x) 
                loss2 = -log(probs[y])          # find new loss
                d_loss = loss2 - loss           # change in loss
                emperical_grad = d_loss/epsilon # approximate emperical gradient
                calc_grad = sen_b[k][index] 
                percent_error = (calc_grad - emperical_grad)/emperical_grad
                b[index] -= epsilon             # return weight to original value
                print str(index) + "\t" + str(calc_grad) + "\t" + str(emperical_grad) + "\t" + str(percent_error)
                F[k][index] = percent_error
        return E, F
                
                
        

    def save(self, file_name):
        pass

def loadPolicyNet(file_name):
    pass


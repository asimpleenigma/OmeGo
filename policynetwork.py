import numpy as np
from numpy import exp, array, zeros, dot, copy, copyto, outer, log
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
        #C, F, h, w = W_s[0].shape
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
        state = self.state
        L = self.L

        
        sen_W = [] # list of weight gradients
        sen_b = [] # bias gradients
        for k in range(L): # initialize sensitivities for weight update
            sen_W += [zeros(W_s[k].shape)]
            sen_b += [zeros(b_s[k].shape)]
        total_loss = 0.
        for i in range(batch_size):
            
            x, y = data_set.sample()
            
            ### Forward Pass ###
            probs  = self.evaluate(x)

            loss = -log(probs[y]) # surprisal
            total_loss += loss

            ### Backward Pass ###
            sen_probs = zeros(state[-1].shape)
            sen_probs[y] = - 1. / probs[y] # surprisal back, see calculus
            
            # softmax back
            sen_scores = -probs * probs[y] # for j != y, because math
            sen_scores[y] = probs[y] * (1 - probs[y]) # because math

            sen_h = sen_scores # sensitivity of the output, upstream gradient
            # k = L-1
            sen_b[L-1] += sen_h
            sen_W[L-1] += convOuter(state[L-1], sen_h, W_s[L-1].shape) # downstream state & upstream grad

            # k = L-2
            sen_h = deconvolve(W_s[L-1], sen_b[L-1]) # conv back, no relu
            for k in range(L-2, -1, -1): # from L-2 to 0
                sen_b[k] += sen_h
                sen_net = (state[k+1]>0.)*sen_h # relu back, sensitivity of net value
                sen_W[k] += convOuter(state[k], sen_net, W_s[k].shape) # downstream state & upstream grad
                sen_h = deconvolve(W_s[k], sen_net) # conv back, sensitivity of downstream layer
                
                

        # Update weights
        for k in range(L):
            W_s[k] += (LR/batch_size) * sen_W[k]
            b_s[k] += (LR/batch_size) * sen_b[k]

        return total_loss / batch_size
        

    def save(self, file_name):
        pass

def loadPolicyNet(file_name):
    pass


import numpy as np
from numpy import zeros, ones, e

__all__ = ['convolve', 'deconvolve', 'convOuter']

def convolve(in_layer, filters, pad_input=True):
    # layer: channel x row x col
    # filters: filter x channel x row x col
    # C: number of channels input layer has
    # n_in: number of rows input layer has
    # m_in: number of columns input layer has
    # F: number of filters, number of output channels
    # h: height of filter
    # w: width of filter
    # n_out: number of rows output has
    # m_out: number of columns output has

    
    F, D, h, w = filters.shape
    C, n_in, m_in = in_layer.shape
    assert (C == D);
    if pad_input:
        assert (h%2==1);
        assert (w%2==1);
        result = zeros(in_layer.shape)
        in_layer = zero_pad(in_layer, h/2, w/2)
        C, n_in, m_in = in_layer.shape
    n_out = (n_in-h) + 1     
    m_out = (m_in-w) + 1
    result = zeros((F, n_out, m_out))
    
    for f in range(F): # for each filter
        for i in range(n_out): # i, j is top-left coordiante of filter placement
            for j in range(m_out):
                x = i
                y = j
                a = in_layer[:, x:x+h, y:y+w] # all channels. rows and cols over filter
                result[f, i, j] = np.sum(a * filters[f])
    return result

def zero_pad(a, pad_h, pad_w): # pads last 2 dims of 3-D array
    f, w, h = a.shape
    result = zeros((f, h+2*pad_h, w+2*pad_w))
    result[:, pad_h:-pad_h, pad_w:-pad_w] = a
    return result
    
def deconvolve(filters, out_layer, trim_inlayer=True):
    # in_layer: channel x row x col
    # filters: filter x channel x row x col
    # C: number of channels input layer has
    # n_in: number of rows input layer has
    # m_in: number of columns input layer has
    # F: number of filters, number of output channels
    # h: height of filter, assumed to be odd
    # w: width of filter, assumed to be odd
    # n_out: number of rows output has
    # m_out: number of columns output has
    
    F, C, h, w = filters.shape
    F2, n_out, m_out = out_layer.shape
    
    assert (F == F2)
    n_in = (n_out-1) + h
    m_in = (m_out-1) + w
    result = zeros((C, n_in, m_in))

    for f in range(F):
        for i in range(n_out):
            for j in range(m_out):
                x = i
                y = j
                result[:, x:x+h, y:y+w] += out_layer[f, i, j] * filters[f] # C x H x W
    if trim_inlayer:
        h_pad = h/2
        w_pad = w/2
        result = result[:, h_pad:-h_pad, w_pad:-w_pad]
    return result

    

def convOuter(in_layer, out_layer, filter_shape=None):
    # in_layer: channel x row x col
    # filters: filter x channel x row x col
    # C: number of channels input layer has
    # n_in: number of rows input layer has
    # m_in: number of columbs input layer has
    # F: number of filters, number of output channels
    # h: height of filter, assumed to be odd
    # w: width of filter, assumed to be odd
    # n_out: number of rows output has
    # m_out: number of columns output has
    if filter_shape: # infer padding from filter shape
        F, C, w, h = filter_shape
        pad1 = w/2 # assume odd
        pad2 = h/2
        in_layer = zero_pad(in_layer, pad1, pad2)
    F, n_out, m_out = out_layer.shape
    C, n_in, m_in = in_layer.shape
        
    h = n_in - 1 * (n_out-1)
    w = m_in - 1 * (m_out-1)
    filters = zeros((F, C, h, w))

    for f in range(F):
        for i in range(n_out):
            for j in range(m_out):
                x = i
                y = j
                filters[f] += out_layer[f, i, j] * in_layer[:, x:x+h, y:y+w]
    return filters
    









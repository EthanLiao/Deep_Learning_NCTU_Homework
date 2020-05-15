import numpy as np


# def sigmoid(x):
#     return 1. / (1.+np.exp(-x))


def softmax(x):
    """
    Caculate the softmax
    exp(x) / sum of each row in exp(x)
    exp(x) is an element wise operator

    Parameters
    -----------
    x : nd-array
        input data can be two or one dimension
    """
    # if x.ndim == 2:
    #     x= x.T # we consider feature of input data is large
    #     x = x - np.max(x,axis=0) # modify each feature in the x
    #     y = np.exp(x) / np.sum(np.exp(x),axis=0)
    #     return y.T # Re-transport
    # x = x-np.max(x)
    # return np.exp(x)/np.sum(np.exp(x))
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y,t):
    """
    Caculate the cross entropy
    =-t*log(y)
    Parameters
    -----------
    t : one-hot-vector
        traing data label

    y : float
        Prediction result of neuron network
    """
    # delta = 1e-7
    # # return -np.sum(t*np.log(y+delta))
    # if y.ndim == 1:
    #     t = t.reshape(1, t.size)
    #     y = y.reshape(1, y.size)
    #
    # if t.size == y.size:
    #     t = t.argmax(axis=1)
    #
    # batch_size = y.shape[0]
    # # t = t.astype(int)
    # return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

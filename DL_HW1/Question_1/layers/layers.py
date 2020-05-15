import numpy as np
from utils.utils import *

"""
Expect the SoftmaxWithLoss,other layers' forward
parameter should only be x and the backward parameter
should be set as dout

"""
class Affine:
    def __init__(self,W,b):
        """
        Set the layers parameters from a dictionary

        Parameters
        -----------
        W : dict
            A dictionary of layer parameters
        b : float
            A bias term for Affine layer
        """
        self.W = W
        self.b = b
        self.X = None
        self.X_original_shape = None
        self.dw = None
        self.db = None
    def forward(self,X):
        """
        Use for network prediction

        Parameters
        ---------
        X : ndarray
            A
        Returns
        ---------
        out :  ndarray
            Affine layers output which is linear
            function of input data
        """
        self.X_original_shape = X.shape
        X = X.reshape(X.shape[0],-1)
        self.X=X
        # print(self.X.shape)
        # print(self.W.shape)
        out = self.X.dot(self.W)+self.b
        return out

    def backward(self,dout):
        """
        Use for network back propagation

        Parameters
        ---------
        dout : ndarray
            Come frome the deriative of activation functions

        Returns
        ---------
        dx :  ndarray
            dx come from the previous layer output
            it should be return value while back propagate
        """
        # print(dout.shape)
        # print(self.W.T.shape)
        dx = dout.dot(self.W.T).reshape(*self.X_original_shape)
        # reshape will ensure the X shape keep the same as input shape
        self.dw = self.X.T.dot(dout)
        self.db = dout.sum(axis=0)
        return dx

class Sigmoid:
    def __init__(self):
        """
        Set the Sigmoid layers
        Parameters
        ------------
        X : ndarray
            input data
        """
        self.y = None

    def forward(self,X):
        """
        Use for network prediction

        Parameters
        ---------
        X : ndarray
            data matrix

        Returns
        ---------
        y :  ndarray
            sigmoid function of input data
        """
        y = sigmoid(X)
        self.y = y
        return y
    def backward(self,dout):
        """
        Use for network back propagation

        Parameters
        ---------
        dout : ndarray
            Come frome the deriative of activation functions

        Returns
        ---------
        dx :  ndarray
            dx come from the previous layer output
            it should be return value while back propagate
        """
        y = self.y
        dx = dout*y*(1.-y)
        return dx

class ReLU:
    def __init__(self):
        """
        Set the layers parameters

        Parameters
        -----------
        self.mask : dict
            A mask indicator indicates position of negtive
            value in an array
        """
        self.mask = None
    def forward(self,X):
        """
        Use for network prediction

        Parameters
        ---------
        X : ndarray
            data matrix

        Returns
        ---------
        y :  ndarray
            ReLU activation function of input data
        """
        self.mask = (X<=0)
        out = X.copy()
        out[self.mask] = 0
        return out

    def backward(self,dout):
        # print(dout.shape)
        # print(self.mask.shape)
        # self.mask = (dout<=0)
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        """
        Set the layers parameters

        Parameters
        -----------
        self.loss : float
            Loss of NN while traing
        self.t : one-hot-vector
        self.y : ndarray
            prediction of NN
        """
        self.loss = None
        self.t = None
        self.y = None
    def forward(self,X,t):
        """
        Use for network prediction

        Parameters
        ---------
        X : ndarray
            X come from the last hidden layer output

        Returns
        ---------
        self.loss :  float
            Loss of NN while traing
        """
        self.t = t
        self.y = softmax(X)
        self.loss = cross_entropy_error(self.y,self.t)
        return self.loss

    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            # print(np.arange(batch_size))
            # print(self.t)
            # print(dx)
            dx[np.arange(batch_size),self.t] -= 1
            dx = dx/batch_size
        return dx

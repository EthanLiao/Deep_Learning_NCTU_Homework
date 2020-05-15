# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
import numpy as np
import csv
from layers.layers import Affine,ReLU,SoftmaxWithLoss
from collections import OrderedDict


class ThreeLayerNet:
    def __init__(self,input_size,hidden_size_1,hidden_size_2,output_size):
        # initialize all the Affine layers
        self.params = {}
        self.params['W1'] = np.random.randn(input_size,hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = np.random.randn(hidden_size_1,hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = np.random.randn(hidden_size_2,output_size)
        self.params['b3'] = np.zeros(output_size)
        # build up the layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'],self.params['b3'])
        self.lastLayer = SoftmaxWithLoss() #lastlayer will not be covered in the dictionary
        # a redundant layer for caculate loss
        # self.losslayer = SoftmaxWithLoss()
        # prediction result
        # self.result = None
        # self.loss_value = None
        self.grad_result = {}

    # this prediction result hasn't passed through softmax yet
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x


    # this prediction result has passed through sigmoid
    def predict_result(self,x,t):
        y=self.predict(x)
        y=self.lastLayer.forward(y,t)
        # print(np.argmax(y,axis=0))
        y=np.argmax(y,axis=0)
        return y
        # accuracy = np.sum(y==t)/ float(x.shape[0])

    def loss(self,x,t):
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def gradient_decent(self,x,t):
        # back propagation
        y = self.predict(x)
        self.lastLayer.forward(y,t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        back_layers = list(self.layers.values())
        back_layers.reverse()
        for layer in back_layers:
            dout = layer.backward(dout)

        # record the params after back prop
        self.grad_result['W1'] = self.layers['Affine1'].dw
        self.grad_result['b1'] = self.layers['Affine1'].db
        self.grad_result['W2'] = self.layers['Affine2'].dw
        self.grad_result['b2'] = self.layers['Affine2'].db
        self.grad_result['W3'] = self.layers['Affine3'].dw
        self.grad_result['b3'] = self.layers['Affine3'].db

    def error_rate(self,x,t):
        y = self.predict_result(x,t)
        return float(sum(y==t))/float(x.shape[0])


# read file and slicing data
all_data = np.zeros(shape=(1,7))
skip_first = True
with open('titanic.csv') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        if skip_first:
            skip_first=False
            continue
        append_array = np.array([float(item) for item in row])
        all_data = np.vstack((all_data,append_array))
    csvfile.close()
    all_data = np.delete(all_data,obj=0,axis=0)

x_data = all_data[:,1:] ; t_data = all_data[:,0] ; t_data = t_data.astype(int)
x_train = x_data[:800,:] ; x_test = x_data[800:891,:]
t_train = t_data[:800] ; t_test = t_data[800:891]

max_iter = 1000
learning_rate = 0.01
batch_size = 20
train_size = x_train.shape[0]
network = ThreeLayerNet(6,3,3,2)
learning_loss_list = [] ; training_error_list = [] ; test_error_list = []

for iter in range(max_iter):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    network.gradient_decent(x_batch,t_batch)
    for key in ['W1','b1','W2','b2','W3','b3']:
        network.params[key] -= learning_rate*network.grad_result[key]
    print(network.loss(x_batch,t_batch))
    learning_loss_list.append(network.loss(x_batch,t_batch))
    training_error_list.append(network.error_rate(x_train,t_train))
    test_error_list.append(network.error_rate(x_test,t_test))

# to do : plot the loss and two error rate

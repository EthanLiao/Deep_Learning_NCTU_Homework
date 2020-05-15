# https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/common/layers.py
import sys, os
sys.path.append(os.pardir)
import numpy as np
from layers.layers import Affine,ReLU,SoftmaxWithLoss
from collections import OrderedDict
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class ThreeLayerNet:
    def __init__(self,input_size,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4,hidden_size_5,output_size):
        # initialize all the Affine layers
        self.params = {}
        self.params['W1'] = 0.1*np.random.randn(input_size,hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = 0.1*np.random.randn(hidden_size_1,hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = 0.1*np.random.randn(hidden_size_2,hidden_size_3)
        self.params['b3'] = np.zeros(hidden_size_3)
        self.params['W4'] = 0.1*np.random.randn(hidden_size_3,hidden_size_4)
        self.params['b4'] = np.zeros(hidden_size_4)
        self.params['W5'] = 0.1*np.random.randn(hidden_size_4,hidden_size_5)
        self.params['b5'] = np.zeros(hidden_size_5)
        self.params['W6'] = 0.1*np.random.randn(hidden_size_5,output_size)
        self.params['b6'] = np.zeros(output_size)
        # Question 1-2
        # self.params['W1'] = 0.1*np.zeros((input_size,hidden_size_1))
        # self.params['b1'] = np.zeros(hidden_size_1)
        # self.params['W2'] = 0.1*np.zeros((hidden_size_1,hidden_size_2))
        # self.params['b2'] = np.zeros(hidden_size_2)
        # self.params['W3'] = 0.1*np.zeros((hidden_size_2,hidden_size_3))
        # self.params['b3'] = np.zeros(hidden_size_3)
        # self.params['W4'] = 0.1*np.zeros((hidden_size_3,hidden_size_4))
        # self.params['b4'] = np.zeros(hidden_size_4)
        # self.params['W5'] = 0.1*np.zeros((hidden_size_4,hidden_size_5))
        # self.params['b5'] = np.zeros(hidden_size_5)
        # self.params['W6'] = 0.1*np.zeros((hidden_size_5,output_size))
        # self.params['b6'] = np.zeros(output_size)
        # build up the layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'],self.params['b3'])
        self.layers['ReLU3'] = ReLU()
        self.layers['Affine4'] = Affine(self.params['W4'],self.params['b4'])
        self.layers['ReLU4'] = ReLU()
        self.layers['Affine5'] = Affine(self.params['W5'],self.params['b5'])
        self.layers['ReLU5'] = ReLU()
        self.layers['Affine6'] = Affine(self.params['W6'],self.params['b6'])
        self.lastLayer = SoftmaxWithLoss() #lastlayer will not be covered in the dictionary


    # this prediction result hasn't passed through softmax yet
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

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

        grad_result = {}
        # record the params after back prop
        grad_result['W1'] = self.layers['Affine1'].dw
        grad_result['b1'] = self.layers['Affine1'].db
        grad_result['W2'] = self.layers['Affine2'].dw
        grad_result['b2'] = self.layers['Affine2'].db
        grad_result['W3'] = self.layers['Affine3'].dw
        grad_result['b3'] = self.layers['Affine3'].db
        grad_result['W4'] = self.layers['Affine4'].dw
        grad_result['b4'] = self.layers['Affine4'].db
        grad_result['W5'] = self.layers['Affine5'].dw
        grad_result['b5'] = self.layers['Affine5'].db
        grad_result['W6'] = self.layers['Affine6'].dw
        grad_result['b6'] = self.layers['Affine6'].db

        return grad_result

    # this prediction result has passed through sigmoid
    def predict_result(self,x,t):
        y=self.predict(x)
        # y=self.lastLayer.forward(y,t)
        # print(np.argmax(y,axis=0))
        y=np.argmax(y,axis=1)
        # print(y.shape)
        return y


    def accuracy_rate(self,x,t):
        y = self.predict_result(x,t)
        t = np.argmax(t,axis=1)
        return float(np.sum(y==t))/float(x.shape[0])
        # y = self.predict(x)
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1 : t = np.argmax(t, axis=1)
        # accuracy = np.sum(y == t) / float(x.shape[0])
        # return accuracy

    def find_key_by_value(self,t_value):
        for (key,value) in self.layers.items():
            if value == t_value:
                return key
    def latent_var(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
            if self.find_key_by_value(layer) ==  'Affine5':
                return x

    def array_2_oneHot(self,a):
        a = a.astype(int)
        b = np.zeros((a.size,a.max()+1))
        b[np.arange(a.size),a]=1
        return b


max_iter = 2000
learning_rate = 0.0001
batch_size = 3000
LATENT = True
latent_var_num = 300
latent_mask = np.array([e for e in range(latent_var_num)])
network = ThreeLayerNet(784,392,196,98,49,24,10)
loss_list = [] ; train_acc_list = [] ; test_acc_list = []

# load data
train_data = np.load('./train/image.npy')
train_data_label = np.load('./train/label.npy')
test_data = np.load('./test/image.npy')
test_label = np.load('./test/label.npy')


partial_x_train = train_data.reshape(train_data.shape[0],-1)
partial_t_train = train_data_label
x_val = test_data.reshape(test_data.shape[0],-1)
y_val = test_label





x_train = partial_x_train ; x_test = x_val
t_train = network.array_2_oneHot(partial_t_train); t_test = network.array_2_oneHot(y_val)
train_size = x_train.shape[0]

for iter in range(max_iter):
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    grad = network.gradient_decent(x_batch,t_batch)
    for key in ['W1','b1','W2','b2','W3','b3','W4','b4','W5','b5','W6','b6']:
        network.params[key] -= learning_rate*grad[key]
    loss_list.append(network.loss(x_batch,t_batch))
    train_acc_list.append(network.accuracy_rate(x_train,t_train))
    test_acc_list.append(network.accuracy_rate(x_test,t_test))
    print('epoch:%.4f' % iter,end = " ")
    print('Loss : %.4f' % network.loss(x_batch,t_batch),end = "　")
    print('training_acc : %.4f' % network.accuracy_rate(x_train,t_train),end = "　")
    print('test_acc : %.4f' % network.accuracy_rate(x_test,t_test))
    if LATENT :
        if iter % 20 ==0 :
            latent_x = x_test[latent_mask]
            latent_y = np.argmax(t_test[latent_mask],axis=1)
            pred_latent_y = network.latent_var(latent_x)
            # scatter params
            colormap = np.array([i for i in range(10)])
            plt.figure()
            plt.scatter(pred_latent_y[:,0],pred_latent_y[:,1],c=colormap[latent_y],alpha=0.5)
            plt.title('latent variable plot epoch: ' + str(iter))
            plt.savefig('latent_variable_plot epoch: ' + str(iter)+'.png')


# plot confusion matrix
confu_mat = np.zeros((10,10))
for x,y in zip(t_test,x_test):
    real_y = np.argmax(x,axis=0)
    pred_y = network.predict_result(y.reshape(1,y.shape[0]),x.reshape(1,x.shape[0]))
    confu_mat[real_y][pred_y[0]]+=1
df_cm = pd.DataFrame(confu_mat, range(10), range(10))
plt.figure(figsize=(20,20))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.title('confusion matrix') ; plt.savefig('confusion_matrix.png')

# plot train, test, loss curve
epoch_list = [i for i in range(max_iter)]
plt.figure();plt.plot(epoch_list,test_acc_list);  plt.title('test accuracy');  plt.savefig('DNN_test_accuracy.png')
plt.figure();plt.plot(epoch_list,train_acc_list); plt.title('train accuracy'); plt.savefig('DNN_train_accuracy.png')
plt.figure();plt.plot(epoch_list,loss_list);      plt.title('losss');          plt.savefig('DNN_loss.png')

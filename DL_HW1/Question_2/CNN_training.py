import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import csv
from PIL import Image
import cv2
import numpy as np


# Hyper parameters
EPOCH = 100
BATCH_SIZE = 50
LR = 0.0001
label_map ={
'bad' : 0,
'good' : 1,
'none' : 2
}

img_dir = './images/'
train_data_dir = './train.csv'
test_data_dir = './test.csv'

def preprocess_image(save_name):
    pil_im = Image.open(img_dir+save_name)
    crop_window = (row['xmin'],row['ymin'],row['xmax'],row['ymax'])
    crop_window = map(int,crop_window)           # convert the value into int
    pil_im = pil_im.crop((crop_window))          # crop image
    open_cv_im = np.array(pil_im.convert('RGB')) # convert the PIL object into cv
    open_cv_im = open_cv_im[:,:,::-1].copy()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(open_cv_im, -1, sharpen_kernel)
    sharpen = cv2.resize(sharpen,(64,64),interpolation=cv2.INTER_CUBIC) # convert the cv2 object into numpy
    # cv2.imwrite('./HW2/copy_images/'+save_name,sharpen)
    trans_image = np.reshape(sharpen,(1,3,64,64))
    return trans_image

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 64,64)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 32, 32)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 32, 32)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 32, 32)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 16, 16)
        )
        self.out = nn.Linear(32 * 16 * 16, 3)   # fully connected layer, output 3 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN().cuda()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# Read Data
train_image_x = np.zeros((1,3,64,64)) ; train_image_t = np.zeros(1)
test_image_x = np.zeros((1,3,64,64)) ; test_image_t = np.zeros(1)

with open(train_data_dir,newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        save_name = row['filename']
        trans_image = preprocess_image(save_name)
        train_image_x = np.concatenate([train_image_x,trans_image],axis=0)
        train_image_t = np.vstack((train_image_t,label_map[row['label']]))

with open(test_data_dir,newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        save_name = row['filename']
        trans_image = preprocess_image(save_name)
        test_image_x = np.concatenate([test_image_x,trans_image],axis=0)
        test_image_t = np.vstack((test_image_t,label_map[row['label']]))

# delete the redundent data
train_image_x = np.delete(train_image_x,0,0) ; test_image_x = np.delete(test_image_x,0,0)
train_image_t = np.delete(train_image_t,0,0) ; test_image_t = np.delete(test_image_t,0,0)

# transfer data into tensor vector form
train_tensor_x = torch.Tensor(train_image_x).cuda()
train_tensor_y = torch.Tensor(train_image_t).cuda()
test_tensor_x = torch.Tensor(test_image_x).cuda()
test_tensor_y = torch.Tensor(test_image_t).cuda()

# For batch normalization , training data should be wrapped into wrapper
train_data = Data.TensorDataset(train_tensor_x,train_tensor_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# training and testing
test_acc_list = [] ; train_acc_list =[] ; loss_list = []
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        output = cnn(b_x)[0]            # cnn output
        b_x = b_x.cuda()
        b_y = b_y.squeeze_().long().cuda()
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        if step % EPOCH == 0 :
            test_output, last_layer = cnn(test_tensor_x)
            train_output, t_last_layer = cnn(train_tensor_x)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            t_pred_y = torch.max(train_output, 1)[1].cpu().data.numpy()
            accuracy = np.sum(pred_y == test_tensor_y.cpu().data.numpy().T.astype('int')) / float(test_tensor_y.size(0))
            t_accuracy = np.sum(t_pred_y == train_tensor_y.cpu().data.numpy().T.astype('int')) / float(train_tensor_y.size(0))
            loss_list.append(loss)
            test_acc_list.append(accuracy)
            train_acc_list.append(t_accuracy)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy, '| train accuracy: %.2f' % t_accuracy)

epoch_list = [i for i in range(EPOCH)]
plt.figure();plt.plot(epoch_list,test_acc_list);plt.title('test accuracy');plt.savefig('test_accuracy.png')
plt.figure();plt.plot(epoch_list,train_acc_list);plt.title('train accuracy');plt.savefig('train_accuracy.png')
plt.figure();plt.plot(epoch_list,loss_list);plt.title('losss');plt.savefig('loss.png')
torch.save(cnn.state_dict(),'./HW2/training_modle2')

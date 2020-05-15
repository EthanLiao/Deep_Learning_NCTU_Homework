import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import csv
from PIL import Image
import cv2
import numpy as np
from CNN import CNN

model_dir = './training_modle'
img_dir = './images/'
train_data_dir = './train.csv'
test_data_dir = './test.csv'

label_map ={
'bad' : 0,
'good' : 1,
'none' : 2
}

# print('sucess')

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
    # cv2.imwrite('./copy_images/'+save_name,sharpen)
    trans_image = np.reshape(sharpen,(1,3,64,64))
    return trans_image


def predict(train_image_x,train_image_t,test_image_x,test_image_t):
    device = torch.device("cpu")
    # cnn = torch.load('./training_modle',map_location = device)
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_dir,map_location = device))
    # cnn.eval()
    # transfer data into tensor vector form
    train_tensor_x = torch.Tensor(train_image_x)
    train_tensor_y = torch.Tensor(train_image_t)
    test_tensor_x = torch.Tensor(test_image_x)
    test_tensor_y = torch.Tensor(test_image_t)

    # prediction
    with torch.no_grad():
        test_output, last_layer = cnn(test_tensor_x)
        train_output, t_last_layer = cnn(train_tensor_x)
        # calculate accuracy
        pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
        t_pred_y = torch.max(train_output, 1)[1].cpu().data.numpy()
        accuracy = np.sum(pred_y == test_tensor_y.cpu().data.numpy().T.astype('int')) / float(test_tensor_y.size(0))
        t_accuracy = np.sum(t_pred_y == train_tensor_y.cpu().data.numpy().T.astype('int')) / float(train_tensor_y.size(0))

        return (accuracy,t_accuracy)


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


good_train_x = np.zeros((1,3,64,64)) ; good_train_y = np.zeros(1)
none_train_x = np.zeros((1,3,64,64)) ; none_train_y = np.zeros(1)
bad_train_x = np.zeros((1,3,64,64)) ; bad_train_y = np.zeros(1)

good_test_x = np.zeros((1,3,64,64)) ; good_test_y = np.zeros(1)
none_test_x = np.zeros((1,3,64,64)) ; none_test_y = np.zeros(1)
bad_test_x = np.zeros((1,3,64,64)) ; bad_test_y = np.zeros(1)

for x,y,m,k in zip(train_image_x,train_image_t,test_image_x,test_image_t):
    x = np.reshape(x,(1,3,64,64))
    m = np.reshape(m,(1,3,64,64))
    if y==0 :
        bad_train_x = np.concatenate([bad_train_x,x],axis=0)
        bad_train_y = np.vstack((bad_train_y,y))
    elif y==1 :
        good_train_x = np.concatenate([good_train_x,x],axis=0)
        good_train_y = np.vstack((good_train_y,y))
    else:
        none_train_x = np.concatenate([none_train_x,x],axis=0)
        none_train_y = np.vstack((none_train_y,y))

    if k==0 :
        bad_test_x = np.concatenate([bad_test_x,m],axis=0)
        bad_test_y = np.vstack((bad_test_y,k))
    elif k==1 :
        good_test_x = np.concatenate([good_test_x,m],axis=0)
        good_test_y = np.vstack((good_test_y,k))
    else:
        none_test_x = np.concatenate([none_test_x,m],axis=0)
        none_test_y = np.vstack((none_test_y,k))

bad_train_x = np.delete(bad_train_x,0,0) ; bad_test_x = np.delete(bad_test_x,0,0)
bad_train_y = np.delete(bad_train_y,0,0) ; bad_test_y = np.delete(bad_test_y,0,0)

good_train_x = np.delete(good_train_x,0,0) ; good_test_x = np.delete(good_test_x,0,0)
good_train_y = np.delete(good_train_y,0,0) ; good_test_y = np.delete(good_test_y,0,0)

none_train_x = np.delete(none_train_x,0,0) ; none_test_x = np.delete(none_test_x,0,0)
none_train_y = np.delete(none_train_y,0,0) ; none_test_y = np.delete(none_test_y,0,0)

print('|bad  test accuracy: %.2f' % predict(bad_train_x,bad_train_y,bad_test_x,bad_test_y)[0], '|bad train accuracy: %.2f' % predict(bad_train_x,bad_train_y,bad_test_x,bad_test_y)[1])
print('|good test accuracy: %.2f' % predict(good_train_x,good_train_y,good_test_x,good_test_y)[0], '|good train accuracy: %.2f' % predict(good_train_x,good_train_y,good_test_x,good_test_y)[1])
print('|none test accuracy: %.2f' % predict(none_train_x,none_train_y,none_test_x,none_test_y)[0], '|none train accuracy: %.2f' % predict(none_train_x,none_train_y,none_test_x,none_test_y)[1])

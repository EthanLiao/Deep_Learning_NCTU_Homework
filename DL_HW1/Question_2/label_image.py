import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import csv
from PIL import Image
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw,ImageEnhance
from CNN import CNN
import numpy as np

img_dir = './images/'
train_data_dir = './problem.csv'
model_dir = './training_modle'
image_name = 'TASS38012699.jpg'

label_map ={
 0 : 'bad',
 1 : 'good',
 2 : 'none'
}

color_map = {
 0 : (12,36,255),
 1 : (36,255,12),
 2 : (255,12,36),
}
pil_image = Image.open(img_dir+image_name).convert('RGB')
cv_image = np.array(pil_image)
cv_image = cv_image[:, :, ::-1].copy()

def preprocess_image(save_name):
    pil_im = Image.open(img_dir+save_name)
    crop_window = (row['xmin'],row['ymin'],row['xmax'],row['ymax'])
    crop_window = map(int,crop_window)                                  # convert the value into int
    pil_im = pil_im.crop((crop_window))                                 # crop image
    open_cv_im = np.array(pil_im.convert('RGB'))                        # convert the PIL object into cv
    open_cv_im = open_cv_im[:,:,::-1].copy()
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(open_cv_im, -1, sharpen_kernel)
    sharpen = cv2.resize(sharpen,(64,64),interpolation=cv2.INTER_CUBIC) # convert the cv2 object into numpy
    trans_image = np.reshape(sharpen,(1,3,64,64))
    return trans_image


def predict(train_image_x):
    device = torch.device("cpu")
    cnn = CNN()
    cnn.load_state_dict(torch.load(model_dir,map_location = device))
    # transfer data into tensor vector form
    train_tensor_x = torch.Tensor(train_image_x)
    # prediction
    with torch.no_grad():
        train_output, t_last_layer = cnn(train_tensor_x)
        # calculate accuracy
        pred_y = torch.max(train_output, 1)[1].cpu().data.numpy()
        return pred_y

with open(train_data_dir,newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        save_name = row['filename']
        trans_image = preprocess_image(save_name)
        classes = predict(trans_image)
        x = int(row['xmin']);y = int(row['ymin']);w = int(row['xmax']);h = int(row['ymax'])
        cv_image= cv2.rectangle(cv_image,(x, y),(w,h),color_map[classes[0]], 1)
        cv2.putText(cv_image, label_map[classes[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_map[classes[0]], 2)
        cv2.imwrite('./test.jpg',cv_image)

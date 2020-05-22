import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import matplotlib.pyplot as plt

train_data = np.load('./train/image.npy')
train_data_label = np.load('./train/label.npy')
test_data = np.load('./test/image.npy')
test_label = np.load('./test/label.npy')


partial_x_train = train_data.reshape(train_data.shape[0],-1)
partial_t_train = train_data_label
x_val = test_data.reshape(test_data.shape[0],-1)
y_val = test_label

model = models.Sequential()
model.add(layers.Dense(392,activation='relu',input_shape=(784,)))
model.add(layers.Dense(196,activation='relu'))
model.add(layers.Dense(98,activation='relu'))
model.add(layers.Dense(49,activation='relu'))
model.add(layers.Dense(24,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# sparse_categorical_crossentropy
# print('sucess')
history = model.fit(partial_x_train,
                    partial_t_train,
                    epochs = 500,
                    batch_size = 1000,
                    validation_data = (x_val,y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training_loss')
plt.plot(epochs,val_loss,'b',label='Validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LOSS.png')
plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation_acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.savefig('ACC.png')
model.save('voice_guide_dog.h5')

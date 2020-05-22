from PIL import Image
import numpy as np
import glob,os,math
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from VAE import VAE

# Preprocessing data
filename = "/home/mint/Desktop/Data_Set/Deep_Learning_NCTU_Homework/DL_HW2/data/*.png"

size = (32,32)
train_data_num = 100
shuf_arr = np.arange(train_data_num)
np.random.shuffle(shuf_arr)
data_arr = np.zeros((train_data_num,32,32,3),dtype = float)

for i,file in enumerate(glob.glob(filename)):
    im = Image.open(file)       # useful method to open image
    im = im.convert('RGB')
    im = im.resize(size)
    data_arr[shuf_arr[i],:,:,:] = im
    if i == train_data_num-1:
        break

data_arr = data_arr / 225 #normalize the data

# Training parameters
EPOCHS = 300
BATCH = 250
NUM_BATCH = int(data_arr.shape[0] / BATCH)
LR = 0.001

with tf.device('/GPU:0'):
    # Training Process
    tf.reset_default_graph()
    x_in = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3], name='X')
    y = tf.placeholder(dtype=tf.float32, shape=[None,32,32,3], name='Y')
    y_flat = tf.reshape(y, shape=[-1,32*32*3])
    keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
    vae = VAE()
    sampled_z,latent_result,half_result = vae.encoder(x_in, keep_prob)
    dec = vae.decoder(sampled_z, keep_prob)
    # Define VAE Loss
    flatten_dec = tf.reshape(dec, [-1, 32*32*3])
    mse_loss = tf.reduce_sum(tf.squared_difference(flatten_dec, y_flat), axis=1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * half_result - tf.square(latent_result) - tf.exp(2.0 * half_result), axis=1)
    loss = tf.reduce_mean(mse_loss + latent_loss)
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)
    # Gathering Observation indices
    loss_arr = np.zeros(EPOCHS)
    x_batch = data_arr[:,:,:,:]
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(EPOCHS):
            epoch_loss = 0
            for b in range(NUM_BATCH):
                x_batch_tmp = x_batch[b*BATCH:(b+1)*BATCH,:,:,:]
                # x_batch = data_arr[b*BATCH:(b+1)*BATCH,:,:,:]
                _, c = sess.run([optimizer, loss],feed_dict={x_in:x_batch, y:x_batch, keep_prob:0.8})
                epoch_loss += c
            loss_arr[epoch] = epoch_loss / NUM_BATCH
            print("EPOCH : ", epoch, "LOSS : ", loss_arr[epoch])

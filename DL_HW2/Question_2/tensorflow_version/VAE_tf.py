import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class VAE(object):
    def __init__(self, amt_channel = 3, x_shape = [32,32,3]):

        self.n_latent = 128
        self.inputs_decoder = int(36*amt_channel/2)
        self.reshape_dim = [-1,6,6, amt_channel]
        self.x_shape = x_shape

    def lrelu(self,x,alpha = 0.2):
        return tf.maximum(x, tf.multiply(x, alpha))

    def encoder(self, x_in, keep_prob):
        x = tf.reshape(x_in, shape = [-1,32,32,3])
        x = tf.layers.conv2d(x, filters = 64, kernel_size = 4,strides = 2, padding = 'SAME', activation = self.lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters = 64, kernel_size = 4,strides = 2, padding = 'SAME', activation = self.lrelu)
        x = tf.nn.dropout(x,keep_prob)
        x = tf.layers.conv2d(x, filters = 64, kernel_size = 4,strides = 2, padding = 'SAME', activation = self.lrelu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.flatten(x)
        # last layer is a dense layer to get latent result
        latent_result = tf.layers.dense(x, units = self.n_latent)
        half_result = 0.5 * latent_result
        # get latent variable noise
        noise = tf.random_normal(tf.stack([tf.shape(x)[0], self.n_latent]))
        # get latent variable z
        z = latent_result + tf.multiply(noise, tf.exp(half_result))
        return z,latent_result,half_result

    def decoder(self, sampled_z, keep_prob) :
        x = tf.layers.dense(sampled_z, units = self.inputs_decoder, activation = self.lrelu)
        x = tf.layers.dense(x, units = self.inputs_decoder*2 , activation = self.lrelu)
        x = tf.reshape(x, self.reshape_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
        x = tf.nn.dropout(x,keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
        x = tf.nn.dropout(x,keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu)
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x, units=np.prod(self.x_shape), activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1]+self.x_shape)
        # shape is [-1] + [32,32,3] = [-1,32,32,3]
        return img

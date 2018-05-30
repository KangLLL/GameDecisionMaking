import config
import numpy as np
import tensorflow as tf

settings = tf.app.flags.FLAGS

# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_variable(weight_shape):
    input_channels = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
    return weight, bias

def _conv_variable(weight_shape):
    w = weight_shape[0]
    h = weight_shape[1]
    input_channels = weight_shape[2]
    output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]
    weight = tf.Variable(tf.random_uniform(weight_shape, minval=-d, maxval=d))
    bias = tf.Variable(tf.random_uniform(bias_shape, minval=-d, maxval=d))
    return weight, bias

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")

def build_conv_network():
    # network weights
    W_conv1, b_conv1 = _conv_variable([8, 8, 4, 16])  # stride=4
    W_conv2, b_conv2 = _conv_variable([4, 4, 16, 32])  # stride=2

    W_fc1, b_fc1 = fc_variable([2592, 256])

    # input layer
    s = tf.placeholder(tf.float32, [None, 84, 84, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(_conv2d(s, W_conv1, 4) + b_conv1)
    h_conv2 = tf.nn.relu(_conv2d(h_conv1, W_conv2, 2) + b_conv2)
    h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    return s, h_fc1

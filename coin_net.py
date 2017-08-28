import numpy as np
import pandas as pd
import time
import tensorflow as tf
import sys
import os
import random

from logger import Logging


def coin_training():



    # input variables - image, specs
    img_height = 28
    img_width = 28
    channels = 1
    conv1_h = 5
    conv1_w = 5
    conv1_inputs = 1
    conv1_outputs = 32
    conv2_h = 5
    conv2_w = 5
    conv2_outputs = 64
    fc1_outputs = 1024
    outputs = 10

    # setup placeholder x, y
    x = tf.placeholder(tf.float32, [None, img_height * img_width])
    y_ = tf.placeholder(tf.float32, [None, outputs])
    # setup x, W, b, y
    x_image = tf.reshape(x, [-1, img_height, img_width, channels])
    # setup convolution1, maxpool1
    W_conv1 = tf.Variable(tf.truncated_normal([conv1_h, conv1_w, conv1_inputs, conv1_outputs], stddev=0.1))
    b_conv1 = tf.Variable(tf.Constant(0.1, shape=[conv1_outputs]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') +b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
    )
    # setup convolution2, maxpool2
    W_conv2 = tf.Variable(tf.truncated_normal([conv2_h, conv2_w, conv1_outputs, conv2_outputs], stddev=0.1))
    b_conv2 = tf.Variable(tf.Constant(0.1, shape=[conv2_outputs]))
    h_conv2 = tf.nn.relu(conv2d(x_image, W_conv1) +b_conv1)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')

    # setup fully connected layer and dropout
    W_fc1 = tf.Variable(tf.truncated_normal([(img_height / 4) * (img_width / 4) * conv2_outputs, fc_outputs]))
    b_fc1 = tf.Variable(tf.Constant(0.1, shape=fc1_outputs))

    pool_2_flat = tf.reshape(h_pool2, [-1, (img_height / 4) * (img_width / 4) * conv2_outputs])
    h_fc1 = tf.nn.relu(tf.matmul(pool_2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 =tf.Variable(tf.truncated_normal([fc1_outputs, outputs]))
    b_fc2 = tf.Variable(tf.Constant(0.1, shape=outputs))

    # output y_conv
    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2
    # define loss function


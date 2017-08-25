# author: steve schmidt
# class for neural network build parameters
# coded for game_player, extendable

import gym
import numpy as np
import tensorflow as tf
import time

class Network(object):

    def __init__(self, params={}, test=1):
        self.parameters = params
        if test:
            #input map [-1 H,W,C]
            self.height = 88
            self.width = 80
            self.channels = 1
            self.arch = ['CONV', 'POOL', 'CONV', 'POOL', 'FULL']
            # 3 convolutional layers -> number of convolutions in each
            self.conv_n_maps = [32, 64]
            # size of each kernel must be mxn and equal size list
            self.conv_kernel_sizes = [(5,5),(5,5)]
            # equal size list for strides in each
            self.conv_strides = [1,1]
            # padding for each kernel
            self.conv_paddings = ['SAME', 'SAME']
            # activation function for each layer
            self.conv_activation = [tf.nn.relu, tf.nn.relu]
            # hidden layer dimensions
            self.n_hidden_in = 64 * 7 * 7
            # fully connected inputs
            self.n_hidden = 1024
            # fully connected activation
            self.hidden_activation = tf.nn.relu
            # gym action space for ms pacman, similar games
            #self.env = gym.make('MsPacman-v0')
            #self.obs = self.env.reset()
            self.n_outputs = 1 # there are 9 actions available in default env
            # weight initialization
            self.initializer = tf.contrib.layers.variance_scaling_initializer()

    def deep_network(self, X_state, name):
        prev_layer = X_state
        conv_layers = []
        with tf.variable_scope(name) as scope:
            # loop through params for convolutional layer sizes to pack into list
            for idx, layer in enumerate(self.arch):
                if layer == 'CONV':
                    continue
                if layer = 'POOL':
                    continue
                if layer == 'FULL':
                    continue
            for n_maps, kernel_size, stride, padding, activation in zip(self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides, self.conv_paddings, self.conv_activation):
                if
                prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size, strides=stride, padding=padding, activation=activation, kernel_initializer=self.initializer)
                conv_layers.append(prev_layer)
            # flatten out last convolutional layer to serve input dims to fully connected layer
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, self.n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden, activation=self.hidden_activation, kernel_initializer=self.initializer)
            # configure output layer hidden->output
            outputs = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=self.initializer)
        # pack all of the tf layers into a collection to be run
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = { var.name[len(scope.name):]: var for var in trainable_vars}
        print(outputs)
        for i in conv_layers:
            print(i.shape)
        print(hidden.shape)
        print(outputs.shape)
        for t in trainable_vars_by_name.items():
            print(t)
        return outputs, trainable_vars_by_name

def preprocess_observation(obs, mspacman_color):
    img = obs[1:176:2, :: 2]
    img = img.mean(axis=2)
    img[img==mspacman_color] = 0
    img = (img - 128) / 128 -1
    return img.reshape(88,80,1)

if __name__ == '__main__':
    # test gym environment w ms pacman
    env = gym.make('MsPacman-v0')
    obs = env.reset()
    print(obs.shape)
    mspacman_color = np.array([201,164,74]).mean()
    obs = preprocess_observation(obs, mspacman_color)
    print(obs.shape)
    # test network configuration by printing out tensor dimensions
    nn = Network()
    x_inputs = tf.placeholder(tf.float32, shape=[None, 88, 80, 1])
    nn.q_network(x_inputs, 'test')



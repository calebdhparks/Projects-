# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24,8,8,4)
        self.conv2 = Conv1D(8,16,1,1)
        self.conv3 = Conv1D(16,4,1,1)
        self.layers = [self.conv1,ReLU(),self.conv2,ReLU(),self.conv3,Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights
        #print("weights for simple scan",w1.shape,w2.shape,w3.shape)
	#transpose stuff and reshape
        newW1=np.transpose(w1)
        newW1=np.reshape(newW1,(self.conv1.out_channel,self.conv1.kernel_size,self.conv1.in_channel))
        newW1=np.transpose(newW1, (0, 2, 1))
        #print(newW1.shape)
        newW2 = np.transpose(w2)
        newW2 = np.reshape(newW2, (self.conv2.out_channel, self.conv2.kernel_size, self.conv2.in_channel))
        newW2 = np.transpose(newW2, (0, 2, 1))
        #print(newW2.shape)
        newW3 = np.transpose(w3)
        newW3 = np.reshape(newW3, (self.conv3.out_channel, self.conv3.kernel_size, self.conv3.in_channel))
        newW3 = np.transpose(newW3, (0, 2, 1))
        #print(newW3.shape)
        self.conv1.W = newW1
        self.conv2.W = newW2
        self.conv3.W = newW3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 2, 2, 2)
        self.conv2 = Conv1D(2, 8, 2, 2)
        self.conv3 = Conv1D(8, 4, 2, 1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        #print(w1.shape,w2.shape,w3.shape)
        newW1Filter1=np.transpose(w1)[0,0:48]
        newW1Filter2 = np.transpose(w1)[1, 0:48]
        newW1=np.vstack((newW1Filter1,newW1Filter2))
        newW1 = np.reshape(newW1, (self.conv1.out_channel, self.conv1.kernel_size, self.conv1.in_channel))
        newW1 = np.transpose(newW1, (0, 2, 1))
        listWeights=[]
        for x in range(0,8):
            listWeights.append(np.transpose(w2)[x,0:4])
        newW2=listWeights[0]
        for x in range(1,len(listWeights)):
            newW2=np.vstack((newW2,listWeights[x]))
        newW2 = np.reshape(newW2, (self.conv2.out_channel, self.conv2.kernel_size, self.conv2.in_channel))
        newW2 = np.transpose(newW2, (0, 2, 1))
        newW3 = np.transpose(w3)
        newW3 = np.reshape(newW3, (self.conv3.out_channel, self.conv3.kernel_size, self.conv3.in_channel))
        newW3 = np.transpose(newW3, (0, 2, 1))
        self.conv1.W = newW1
        self.conv2.W = newW2
        self.conv3.W = newW3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

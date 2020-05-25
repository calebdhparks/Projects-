"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        self.logits=[]
        self.current_batch_loss=[]
        self.output=[]
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        if not hiddens:
            layer0=Linear(input_size,output_size,weight_init_fn,bias_init_fn)
            self.linear_layers=[layer0]
        elif len(hiddens)==1:
            self.linear_layers = []
            self.linear_layers.append(Linear(input_size, hiddens[0], weight_init_fn, bias_init_fn))
            self.linear_layers.append(Linear(hiddens[0], output_size, weight_init_fn, bias_init_fn))
        else:
            self.linear_layers = []
            self.linear_layers.append(Linear(input_size, hiddens[0], weight_init_fn, bias_init_fn))
            for i in range(0,len(hiddens)-1):
                in_size=hiddens[i]
                out_size=hiddens[i+1]
                self.linear_layers.append(Linear(in_size, out_size, weight_init_fn, bias_init_fn))
            self.linear_layers.append(Linear(out_size, output_size, weight_init_fn, bias_init_fn))
        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = []
            for i in range(0,self.num_bn_layers):
                self.bn_layers.append(BatchNorm(self.linear_layers[i].W.shape[1]))
                # print('infeatures: ',self.linear_layers[i].W.shape[0])


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        if self.bn:
            input = x
            counter=0
            for i in range (0,self.num_bn_layers):
                out = self.linear_layers[i](input)
                out2 = self.bn_layers[i](out,self.train_mode)
                out3 = self.activations[i](out2)
                input = out3
                counter+=1
            for i in range(counter, len(self.linear_layers)):
                out = self.linear_layers[i](input)
                out2 = self.activations[i](out)
                input = out2
            self.logits = out2
        else:
            input=x
            for i in range(0,len(self.linear_layers)):
                out=self.linear_layers[i](input)
                out2=self.activations[i](out)
                input=out2
            self.logits=out2
        self.output=self.logits
        return self.logits
    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            self.linear_layers[i].dW.fill(0.0)
            self.linear_layers[i].db.fill(0.0)
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].dgamma.fill(0.0)
                self.bn_layers[i].dbeta.fill(0.0)
    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            if self.momentum==0.0:
                self.linear_layers[i].update(self.lr)
            else:
                self.linear_layers[i].update_momentum(self.lr,self.momentum)
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].update(self.lr)
        # raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        # print(self.linear_layers[0].x.shape,labels.shape)
        if self.bn:
            loss = self.criterion(self.logits, labels)
            grad = self.criterion.derivative()
            input = grad
            for i in range(len(self.linear_layers) - 1, -1, -1):
                if i>=self.num_bn_layers:
                    out = np.multiply(input, self.activations[i].derivative())
                    out2 = self.linear_layers[i].backward(out)
                    input = out2
                else:
                    # print("here",i,self.num_bn_layers)
                    out = np.multiply(input, self.activations[i].derivative())
                    out2=self.bn_layers[i].backward(out)
                    out3=self.linear_layers[i].backward(out2)
                    # print("A:",out2.shape,out2,"B:",out3.shape)
                    input=out3
        else:
            loss=self.criterion(self.logits,labels)
            grad=self.criterion.derivative()
            input =grad
            for i in range(len(self.linear_layers)-1,-1,-1):
                out=np.multiply(input,self.activations[i].derivative())
                out2=self.linear_layers[i].backward(out)
                input=out2
        self.current_batch_loss=loss
    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val
    idxs = np.arange(len(trainx))
    idxsVal=np.arange(len(valx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        # Per epoch setup ...
        print("Epoch: ", e,"Completion: ",(e/nepochs*100),"%")
        batchLoss = []
        batchAcc = []
        batchLossVal=[]
        batchAccVal=[]
        np.random.shuffle(idxs)
        np.random.shuffle(idxsVal)
        # print(trainx.shape)
        for b in range(0, len(trainx), batch_size):
            # Train ...
            mlp.train()
            batchIndex=idxs[b:b+batch_size]
            batchX,batchY=setup_arrays(batchIndex,trainx,trainy)
            mlp.zero_grads()
            forwardOut=mlp.forward(batchX)
            mlp.backward(batchY)
            mlp.step()
            batchLoss.append(mlp.total_loss(batchY))
            batchAcc.append(mlp.error(batchY))
        for b in range(0, len(valx), batch_size):
            # Val ...
            mlp.eval()
            batchIndex = idxsVal[b:b + batch_size]
            batchX, batchY = setup_arrays(batchIndex, valx, valy)
            forwardOut = mlp.forward(batchX)
            batchAccVal.append(mlp.error(batchY))
            batchLossVal.append(mlp.total_loss(batchY))



        # Accumulate data...
        training_losses[e]=np.mean(np.asarray(batchLoss))/batch_size
        training_errors[e]=np.mean(np.asarray(batchAcc))/batch_size
        validation_losses[e]=np.mean(np.asarray(batchLossVal))/batch_size
        validation_errors[e]=np.mean(np.asarray(batchAccVal))/batch_size
        # print(validation_losses,validation_errors)
        # print(training_losses,training_errors)
    # Cleanup ...

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)

    # raise NotImplemented
def setup_arrays(idxs,x,y):
    return np.take(x,idxs,axis=0),np.take(y,idxs,axis=0)

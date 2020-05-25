# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):

        """
        Argument:
            W (np.array): (in feature, out feature)
            dW (np.array): (in feature, out feature)
            momentum_W (np.array): (in feature, out feature)

            b (np.array): (1, out feature)
            db (np.array): (1, out feature)
            momentum_B (np.array): (1, out feature)
        """

        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.x=np.zeros(None)
        # TODO: Complete these but do not change the names.
        self.dW = np.zeros(None)
        self.db = np.zeros(None)

        self.momentum_W = np.zeros((in_feature,out_feature))
        self.momentum_b = np.zeros(out_feature)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x=x
        return np.dot(x,self.W)+self.b

    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        b_row,b_col=self.b.shape
        # print(self.b,self.b.shape)
        size=self.x.shape[0]
        delta_b=[]
        # print(delta)
        for i in range(0,delta.shape[1]):
            delta_b.append(np.mean(delta[:,i]))
        delta_b=np.asarray(delta_b)
        delta_w=np.matmul(self.x.transpose(),delta)
        self.dW=delta_w/size
        self.db=delta_b
        self.db=np.reshape(self.db,(b_row,b_col))
        outGrad=np.matmul(delta,self.W.transpose())
        return outGrad
    def update(self,lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db
    def update_momentum(self,lr,momentum):
        self.momentum_W=momentum*self.momentum_W-lr*self.dW
        self.W+=self.momentum_W
        self.momentum_b = momentum * self.momentum_b - lr * self.db
        self.b += self.momentum_b
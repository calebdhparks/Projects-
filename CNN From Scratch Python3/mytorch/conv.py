# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import sys


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        self.x=x
        #print(self.in_channel,self.out_channel,self.kernel_size)
        #print(x.shape,self.W.shape,self.b.shape)
        out_size=((x.shape[2] - 1 * (self.kernel_size - 1) - 1) // self.stride) + 1
        #print(out_size)
        out=np.zeros((x.shape[0],self.out_channel,out_size))
        #print(out.shape)
        #print(self.b)
        for b in range(0, out.shape[0]):
            for c in range(0, self.out_channel):
                window=0
                for y in range(0, out_size):
                   #print(x[:, w:w + self.out_channel, y:y + self.out_channel].shape)
                    out[b, c, y] = np.sum(x[b,:, window:window + self.kernel_size] * self.W[c]) + self.b[c]
                    window+=self.stride
        #print(self.W[0].shape)

        self.out=out
        return out



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        #j=13
        #i=5
        #Dl-1 =5
        batch_size=delta.shape[0]
        layer_size=delta.shape[1]
        self.dx=np.zeros(self.x.shape)
        #print("delta shape",delta.shape,"X shape",self.x.shape,"dW shape",self.dW.shape)
        for j in range (0, delta.shape[1]):
            for outPos in range(0, delta.shape[2]):
                inPos=outPos*self.stride
                for inChannel in range(0, self.dW.shape[1]):
                    for kernel in range( 0,self.dW.shape[2]):
                        self.dx[:,inChannel,inPos+kernel]+=self.W[j,inChannel,kernel]*delta[:,j,outPos]
                        self.dW[j,inChannel,kernel]+=np.sum(delta[:,j,outPos]*self.x[:,inChannel,inPos+kernel])
                        print(self.W[j,inChannel,kernel]*delta[:,j,outPos])
                        print(self.dx[:,inChannel,inPos+kernel])
                        #print()
                        print(delta[:,j,outPos])
                print(self.dx[:,:,inPos:inPos+self.kernel_size])
                sys.exit()
        for c in range (0,self.db.shape[0]):
            self.db[c]=np.sum(delta[:,c,:])
        #self.dW=delta*self.W
        return self.dx



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape((self.b,self.c*self.w))

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape((self.b,self.c,self.w))

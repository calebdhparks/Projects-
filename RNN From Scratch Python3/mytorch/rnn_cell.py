import numpy as np
from activation import *
import sys
class RNN_Cell(object):
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN cell forward (single time step)

        Input (see writeup for explanation)
        ----------
        x : (batch_size, input_size)
        h : (batch_size, hidden_size)

        Returns
        -------
        h_prime : (batch_size, hidden_size)
        """
        A=np.zeros((x.shape[0],self.hidden_size))
        for a in range(0,x.shape[0]):
            A[a]=np.dot(self.W_ih,x[a])+self.b_ih[:]+np.dot(self.W_hh,h[a])+self.b_hh[:]
        h_prime=self.activation(A)
        return h_prime

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN cell backward (single time step)

        Input (see writeup for explanation)
        ----------
        delta : (batch_size, hidden_size)
        h : (batch_size, hidden_size)
        h_prev_l: (batch_size, input_size)
        h_prev_t: (batch_size, hidden_size)

        Returns
        -------
        dx : (batch_size, input_size)
        dh : (batch_size, hidden_size)
        """
        #U => Input to hidden
        #W => Hidden to Hidden
        batch_size = delta.shape[0]
        # print("Delta shape",delta.shape)
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = self.activation.derivative(state=h) * delta
       # print(dz.shape,self.b_hh.shape)
       #  print("dW.ih shape",self.dW_ih.shape,"dW.hh shape",self.dW_hh.shape)
       #  print("H last layer shape",h_prev_l.shape,"H last time shape",h_prev_t.shape)
       #  print("H this layer and time shape",h.shape,"dZ shape",dz.shape)
        #print(h_prev_t*dz)
        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih +=np.matmul(dz.T,h_prev_l)/batch_size
        self.dW_hh +=np.matmul(dz.T,h_prev_t)/batch_size
        self.db_ih +=np.mean(dz,axis=0)
        self.db_hh +=np.mean(dz,axis=0)

        # 2) Compute dx, dh
        dx = np.matmul(dz,self.W_ih)
        dh = np.matmul(dz,self.W_hh)
        # print("dH shape",dh.shape,"dX shape",dx.shape,"\n")

        # 3) Return dx, dh
        return dx, dh

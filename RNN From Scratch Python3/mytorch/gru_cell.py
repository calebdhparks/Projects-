import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here
        self.z=np.zeros((h,))
        self.r = np.zeros((h,))
        self.h_tilda = np.zeros((h,))
        self.forward_out=np.zeros((h,))
    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h
        #print(self.Wzx.shape)
        #print(x.shape)
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        #print(self.x.shape == (self.d, ))
        zt=np.matmul(self.Wzh,h)+np.matmul(self.Wzx,x)
        self.z=self.z_act(zt)
        rt=np.matmul(self.Wrh,h)+np.matmul(self.Wrx,x)
        self.r=self.r_act(rt)
        inner_ht=np.matmul(self.Wh,np.multiply(self.r,h))+np.matmul(self.Wx,x)
        self.h_tilda=self.h_act(inner_ht)
        h_t=(1-self.z)*h+self.z*self.h_tilda
        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )
        self.forward_out=h_t
        return h_t

        #raise NotImplementedError


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        dx=np.zeros((1,self.d))
        dh=np.zeros((1,self.h))
        dz3=delta*self.h_tilda.T
        dz10=delta*self.z.T
        dz11=delta*self.hidden.T
        dz3-=dz11
        dhlast=delta*(1-self.z).T
        dh+=dhlast
        dz9_5=dz10*self.h_act.derivative().T
        dz7=dz9=dz9_5
        dwh=np.matmul(np.reshape(self.r*self.hidden,(self.h,1)),dz9)
        self.dWh+=dwh.T
        dz8=np.matmul(dz9,self.Wh)
        dz6=dz8*self.hidden.T
        dhlast=dz8*self.r.T
        dh+=dhlast
        dwx=np.matmul(np.reshape(self.x,(self.d,1)),dz7).T
        self.dWx+=dwx
        dxt=np.matmul(dz7,self.Wx)
        dx+=dxt
        dz5_5=dz4=dz5=dz6*self.r_act.derivative().T
        self.dWrx+=np.matmul(np.reshape(self.x,(self.d,1)),dz5).T
        dxt=np.matmul(dz5,self.Wrx)
        dx+=dxt
        #print(self.hidden.shape,dz4.shape)
        self.dWrh+=np.matmul(np.reshape(self.hidden,(self.h,1)),dz4).T
        dhlast=np.matmul(dz4,self.Wrh)
        dh+=dhlast
        dz2_5=dz2=dz1=dz3*self.z_act.derivative().T
        self.dWzx+=np.matmul(np.reshape(self.x,(self.d,1)),dz2).T
        dxt=np.matmul(dz2,self.Wzx)
        dx+=dxt
        self.dWzh+=np.matmul(np.reshape(self.hidden,(self.h,1)),dz1).T
        dhlast=np.matmul(dz1,self.Wzh)
        dh+=dhlast
        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        #raise NotImplementedError

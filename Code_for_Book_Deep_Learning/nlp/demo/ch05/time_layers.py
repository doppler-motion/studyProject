import numpy as np


class Rnn:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.out = None
        self.cache = None

    def forward(self, h_pre, x):
        Wx, Wh, b = self.params
        h_next = np.tanh(np.dot(Wx, x) + np.dot(Wh, h_pre) + b)
        self.cache = (x, h_pre, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_pre, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_pre.T, dt)
        dh_pre = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_pre







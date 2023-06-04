import os, sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件
import numpy as np

from utils.general_func import softmax, cross_entropy_error
from utils.gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z  = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = SimpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)


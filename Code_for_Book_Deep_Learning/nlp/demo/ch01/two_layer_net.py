import numpy as np

from forward_net import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, out_put_size):
        I, H, O = input_size, hidden_size, out_put_size

        # 初始化权重
        W1 = np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = np.random.randn(H, O)
        b2 = np.zeros(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]
        self.loss_layers = SoftmaxWithLoss()

        # 将所有参数整理到列表
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layers.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layers.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int)

    for j in range(CLS_NUM):
        for i in range(N):  # N*j, N*(j+1)):
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

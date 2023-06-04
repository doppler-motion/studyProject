import numpy as np

from .functions import softmax, cross_entropy_error
from .log_config import logger


class MatMul:
    """乘法层"""

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, W.T)
        self.grads[0][...] = dW
        return dx


class Affine:
    """
    **全连接层的变换写为Affine层

    **全连接层的变换类似几何学中的仿射变换，因此命名为Affine
    """

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class Embedding:
    """从权重参数中抽取 单词 ID 对应行（向量）的层"""

    def __init__(self, W):
        """
        初始化
        :param W:权重
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        """抽取对应的层"""
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        # dW[self.idx] = dout  # 不太好的方式，有重复id的时候

        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]

        # 或者
        # np.add.at(dW, self.idx, dout)
        return None


class Softmax:
    def __init__(self):
        self.out = None

    def forward(self, x):
        y = softmax(x)
        self.out = y
        return y

    def backward(self, dout=1):
        dx = self.out * dout
        dxsum = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * dxsum
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # print("y.size: ", self.y.size)
        # print("y.ndim: ", self.y.ndim)

        # 当监督标签是one-hot向量时，改为正确解标签的索引
        if self.y.size == self.t.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout=1):
        dx = dout * (1 - self.out) / self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


if __name__ == "__main__":
    W = np.array(np.random.randn(7, 7))
    logger.info(f"W: {W}")
    Em = Embedding(W)
    W_idx = Em.forward([1, 2, 1, 0])
    logger.info(f"W_idx: {W_idx}")

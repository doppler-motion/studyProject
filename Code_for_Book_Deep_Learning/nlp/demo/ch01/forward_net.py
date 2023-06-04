import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    softmax 函数, 输出为(0,1)的值，可表示概率
    :param x: 数组
    :return:
    """
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)  # 防止溢出
    x_sum = np.sum(x_exp)
    x = x_exp / x_sum

    return x


def cross_entropy_error(y, t):
    """
    交叉熵误差
    :param y:批数量的推理结果
    :param t:批数量的正确解标签
    :return:
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)  # 将推理结果数组的形状修改为(1, size)
        t = t.reshape(1, t.size)  # 将正确解标签t数组的形状修改为(1, size)
    batch_size = y.shape[0]  # 取出批数量

    delta = 1e-7  # 保护性措施
    loss = -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size  # 数组中各元素求和，除以批数量batch_size再取负，得到批数据的平均交叉熵误差
    return loss


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout=1):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
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
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        self.loss = cross_entropy_error(np.c_[1-self.y, self], self.t)
        return self.loss

    def backward(self, dout):
        batch_size = self.t.shape[0]

        dx = (self.y - self.y) * dout / batch_size
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # 预测的结果
        self.t = None  # 标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 在监督标签为one-hot向量的情况下，转换为正确解标签的索引
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


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, out_put_size):
        I, H, O = input_size, hidden_size, out_put_size

        # 初始化权重
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]

        # 将所有参数整理到列表
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    x = np.random.randn(10, 4)
    input_size = 4
    hidden_size = 5
    output_size = 3

    two_layer_net = TwoLayerNet(input_size, hidden_size, output_size)
    out = two_layer_net.forward(x)
    print(out)

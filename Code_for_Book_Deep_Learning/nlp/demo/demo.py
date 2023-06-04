import os, sys

import numpy as np
import matplotlib.pyplot as plt

from common.optimizers import SGD
from common.layers import Affine, SoftmaxWithLoss, Sigmoid

project_path = os.path.dirname(__file__)[
               :os.path.dirname(__file__).index("scripts")] + "/scripts"  # scripts 是我project的名字，根据project要改名字
sys.path.append(project_path)
from common_utils.common_log import logger

arr1 = np.array([[1, 2], [3, 4], [5, 6]])
arr2 = np.array([[10, 20]])
arr3 = arr1 * arr2


# logger.info(arr3)

def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int_)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = rate * 1.0
            theta = 4.0 * j + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        W1 = 0.01 * np.random.randn(input_size, hidden_size)
        b1 = np.zeros(hidden_size)
        W2 = 0.01 * np.random.randn(hidden_size, output_size)
        b2 = np.zeros(output_size)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        self.last_layer = SoftmaxWithLoss()

        # 参数, 梯度添加到同一列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def forward(self, x, t):
        y = self.predict(x)
        # logger.info("y.ndim: ", y.ndim)
        # logger.info("y.size: ", y.size)
        loss = self.last_layer.forward(y, t)
        return loss

    def backward(self, dout=1):
        dout = self.last_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout


def use_two_layer_net():
    # ①设置超参数
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # ②读入数据，生成模型和优化器
    x, t = load_data()
    logger.info(f"x.shape: {x.shape}")
    logger.info(f"t.shape: {t.shape}")
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # ③学习用的参数
    datasize = len(x)
    max_iters = datasize // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # ④打乱数据
        idx = np.random.permutation(datasize)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters * batch_size:(iters + 1) * batch_size]
            batch_t = t[iters * batch_size:(iters + 1) * batch_size]

            # ⑤计算梯度，更新参数
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # ⑥定期输出学习过程
            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                logger.info("| epoch %d |  iter %d / %d | loss %.2f" % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0

    # # 绘制学习结果
    # plt.plot(np.arange(len(loss_list)), loss_list, label='train')
    # plt.xlabel('iterations (x10)')
    # plt.ylabel('loss')
    # plt.show()
    #
    # # 绘制决策边界
    # h = 0.001
    # x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
    # y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # X = np.c_[xx.ravel(), yy.ravel()]
    # score = model.predict(X)
    # predict_cls = np.argmax(score, axis=1)
    # Z = predict_cls.reshape(xx.shape)
    # plt.contourf(xx, yy, Z)
    # plt.axis('on')
    #
    # # 绘制数据点
    # x, t = load_data()
    # N = 100
    # CLS_NUM = 3
    # markers = ['o', 'x', '^']
    # for i in range(CLS_NUM):
    #     plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])
    # plt.show()


if __name__ == "__main__":
    # x_data, t_data = load_data()
    #
    # logger.info(f"x_data: {x_data}")
    # logger.info(f"t_data: {t_data}")
    #
    # net = TwoLayerNet(input_size=10, hidden_size=20, output_size=10)
    # x = np.random.randn(2, 10)
    # # t = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).reshape(1, 10)
    # t = np.array([0, 1])
    # y = net.forward(x, t)
    # logger.info(y)
    #
    # # 输入数据
    # x = np.random.randn(10, 3)
    #
    # # 第一层权重和偏置
    # W1 = np.random.randn(3, 5)
    # b1 = np.random.randn(5)
    #
    # # 第二层权重和偏置
    # W2 = np.random.randn(5, 2)
    # b2 = np.random.randn(2)
    #
    # # 监督数据
    # t = np.array([0, 1])

    # 两层神经网路
    # W1 = np.random.randn(2, 4)
    # b1 = np.random.randn(4)
    # W2 = np.random.randn(4, 5)
    # b2 = np.random.randn(5)
    # x = np.random.randn(10, 2)
    #
    # y1 = np.dot(x, W1) + b1
    # y2 = np.dot(y1, W2) + b2
    # t = sigmoid(y2)
    # logger.info(t)

    use_two_layer_net()

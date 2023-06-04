import numpy as np


from common.functions import sigmoid, identify_function


def init_network():
    """
    进行权重和偏置的初始化
    :return:
    """
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 第1层的权重
    network["B1"] = np.array([0.1, 0.2, 0.3])  # 第1层的偏置

    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])  # 第2 层的权重
    network["B2"] = np.array([0.1, 0.2])  # 第 2 层的偏置

    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])  # 第3层的权重
    network["B3"] = np.array([0.1, 0.2])  # 第3层的偏置

    return network


def forward(network, x):
    """
    将输入信号转为输出信号
    :param network:
    :param x:
    :return:
    """
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["B1"], network["B2"], network["B3"]

    A1 = np.dot(x, W1) + B1  # 第1层输出层的输出
    Z1 = sigmoid(A1)  # 第一层的激活函数
    print(A1)
    print(Z1)

    A2 = np.dot(Z1, W2) + B2  # 第2层的输出
    Z2 = sigmoid(A2)  # 第2层的激活函数
    print(A2)
    print(Z2)

    A3 = np.dot(Z2, W3) + B3  # 第3层的输出
    Y = identify_function(A3)  # 最终输出

    return Y

X = np.array([1.0, 0.5])  # 输入层

network = init_network()

y = forward(network, X)

print(y)

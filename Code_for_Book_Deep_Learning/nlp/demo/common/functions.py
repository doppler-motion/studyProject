import numpy as np

import os, sys

# 加入项目路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.log_config import logger


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cross_entropy_error(y, t):
    """
    mini-batch版本的交叉熵误差函数
    :param y:批数量的推理结果
    :param t: 批数量的正确解标签
    :return:
    """
    if y.ndim == 1:
        y = y.reshape(1, y.size)  # 将推理结果数组的形状修改为(1, size)
        t = t.reshape(1, t.size)  # # 将正确解标签t数组的形状修改为(1, size)
    if y.size == t.size:
        t = np.argmax(t, axis=1)

    batch_size = y.shape[0]  # 取出批数量
    # logger.info("y.shape: ", y.shape)
    # logger.info("batch_size: ", batch_size)
    #
    # logger.info(f"t: {t}")
    # logger.info(f"y: {y}")
    y1 = y[np.arange(batch_size), t]
    # logger.info(f"y1: {y1}")

    # return -np.sum(np.log(y1 + 1e-7)) / batch_size
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  # 数组中各元素求和，除以批数量batch_size再取负，得到批数据的平均交叉熵误差


def softmax(x):
    """
    解决softmax函数的溢出问题，利用c（输入的最大值），softmax函数减去这个最大值保证数据不溢出，softmax函数运算时加上或者
    减去某个常数并不会改变运算的结果
    :param x: 数组
    :return:
    """
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=1, keepdims=True)
        y = x_exp / x_sum
        return y
    elif x.ndim == 1:
        x = x - np.max(x)
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp)
        y = x_exp / x_sum
        return y


if __name__ == "__main__":
    data = np.array([1, 2, 3])
    data1 = np.array([[1, 2, 3], [4, 5, 6]])
    result = sigmoid(data)
    logger.info(result)
    result = softmax(data)
    logger.info(result)
    result = softmax(data1)
    logger.info(result)
    logger.info(np.max(data1, axis=1, keepdims=True))
    logger.info(np.sum(data1, axis=1, keepdims=True))
    logger.info(data1.ndim)
    logger.info(data.ndim)

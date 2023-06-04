import os, sys

import numpy as np

from negative_simpling_layer import NegativeSimplerLoss

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.layers import Embedding
from common.log_config import logger


class CBOW:
    def __init__(self, vocab_size, hidden_size, windows_size, corpus):
        """
        初始化
        :param vocab_size: 单词id个数
        :param hidden_size: 隐藏层的大小
        :param windows_size: 窗口大小
        :param corpus: 单词id列表
        """
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(V, H).astype("f")

        # 生成层
        self.in_layers = []
        for i in range(2 * windows_size):
            layer = Embedding(W_in)  # 使用Embedding层
            self.in_layers.append(layer)

        self.ns_loss = NegativeSimplerLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有权重整理到列表中
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示变为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])  # 根据idx取出特定层
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)

        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

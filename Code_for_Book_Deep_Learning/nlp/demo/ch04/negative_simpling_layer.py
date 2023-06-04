import os, sys
import collections
import numpy as np
import pylab as p

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.log_config import logger
from common.layers import Embedding, SigmoidWithLoss


class EmbeddingDot:
    """取出特定层，并计算内积"""
    def __init__(self, W):
        """
        初始化
        :param W: 输出侧的权重
        """
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnipramSimpler:
    """
    负采样的采样方式
    对概率分布进行幂次运算
    对目标词选取sample_size个数的负例
    """

    def __init__(self, corpus, power, sample_size):
        """
        初始化
        :param corpus: 单词id列表
        :param power: 指数
        :param sample_size: 样本大小
        """
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        """针对目标词，获取负例"""
        batch_size = target.shape[0]

        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        return negative_sample


class NegativeSimplerLoss:
    """负采样"""

    def __init__(self, W, corpus, power=0.75, sample_size=5):
        """
        初始化
        :param W: 输出侧的权重
        :param corpus: 单词id列表
        :param power: 指数
        :param siample_size: 样本大小
        """
        self.sample_size = sample_size
        self.sampler = UnipramSimpler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        # logger.info(batch_size)
        negative_sample = self.sampler.get_negative_sample(target)
        # logger.info(negative_sample)

        # 正例的正向传播
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 负例的正向传播
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[i + 1].forward(h, negative_target)
            loss += self.loss_layers[i + 1].forward(score, negative_label)

        return loss

    def backward(self, dout):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


if __name__ == "__main__":
    W = np.array(np.arange(21).reshape(7, 3))  # 权重
    em_dot = EmbeddingDot(W)  # 初始化模型
    h = np.array([[0, 1, 2], [3, 4, 5]])  #
    idx = [1, 2]  #
    out = em_dot.forward(h, idx)  #
    logger.info(f"out： {out}")

    # use UnipramSimpler
    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2
    sampler = UnipramSimpler(corpus, power, sample_size)
    target = np.array([1, 3, 0])
    negetaive_sample = sampler.get_negative_sample(target)
    logger.info(negetaive_sample)

import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.log_config import logger
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    """
    简单版的CBOW模型
    """

    def __init__(self, vocab_size, hidden_size):
        """
        初始化
        :param vocab_size: 单词个数
        :param hidden_size: 隐藏层的个数
        """
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype("f")
        W_out = 0.01 * np.random.randn(H, V).astype("f")

        # 生成层
        self.layer_in0 = MatMul(W_in)
        self.layer_in1 = MatMul(W_in)
        self.layer_out = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度整理到列表中
        layers = [self.layer_in0, self.layer_in1, self.layer_out]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示变为实例变量
        self.word_vecs = W_in
        logger.info("init simple_cbow success!")

    def forward(self, contexts, target):
        h0 = self.layer_in0.forward(contexts[:, 0])
        h1 = self.layer_in1.forward(contexts[:, 1])
        # logger.info(f"h0: {h0}")
        # logger.info(f"h1: {h1}")
        h = 0.5 * (h0 + h1)
        score = self.layer_out.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.layer_out.backward(ds)
        da *= 0.5
        self.layer_in0.backward(da)
        self.layer_in1.backward(da)
        return None

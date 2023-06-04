import os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.log_config import logger
from common.layers import MatMul

# 样本的上下文数据
c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])

# 权重的初始值
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 生成层
layer_in0 = MatMul(W_in)
layer_in1 = MatMul(W_in)
layer_out = MatMul(W_out)

# 正向传播
h0 = layer_in0.forward(c0)
h1 = layer_in1.forward(c1)
h = 0.5 * (h0 + h1)
out = layer_out.forward(h)
logger.info(f"out: {out}")

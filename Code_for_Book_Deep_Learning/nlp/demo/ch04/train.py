import sys, os
import numpy as np
import pickle

from cbow import CBOW
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.trainers import Trainer
from common.optimizers import Adam
from common.data import load_data
from common.utils import create_contexts_targets, process # to_cpu, to_gpu
from common.log_config import logger

# 设定超参数
windows_size = 1
hidden_size = 3
batch_size = 1
max_epoch = 10
# windows_size = 5
# hidden_size = 100
# batch_size = 100
# max_epoch = 10

text = "you say goobye and i say hello."

# 读入数据
corpus, word_to_id, id_to_word = process(text)
# corpus, word_to_id, id_to_word = load_data("train")
vocab_size = len(corpus)

contexts, target = create_contexts_targets(corpus, windows_size)
logger.info(f"contexts: {contexts}, contexts.shape : {contexts.shape}")
logger.info(f"target: {target}, target size: {len(target)}")

W = np.random.randn(vocab_size, hidden_size)
logger.info(f"W: {W}")
logger.info(f"contexts[:, 0]: {contexts[:, 0]}")
logger.info(f"contexts[:, 1]: {contexts[:, 1]}")
logger.info(f"W[contexts[:, 0]]: {W[contexts[:, 0]]}")
logger.info(f"W[contexts[:, 1]]: {W[contexts[:, 1]]}")
logger.info(f"W[contexts[:, 0]] + W[contexts[:, 1]]: {W[contexts[:, 0]] + W[contexts[:, 1]]}")
logger.info(f"(W[contexts[:, 0]] + W[contexts[:, 1]]) / 2: {(W[contexts[:, 0]] + W[contexts[:, 1]]) / 2}")
logger.info(f"(W[contexts[:, 0]] + W[contexts[:, 1]]) / 2")

# # 生成模型
# model = CBOW(vocab_size, hidden_size, windows_size, corpus)
#
# optimizer = Adam()
# trainer = Trainer(model, optimizer)
#
# # 开始学习
# trainer.fit(contexts, target, max_epoch, batch_size)
# trainer.plot()

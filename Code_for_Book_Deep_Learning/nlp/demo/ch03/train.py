import os.path
import sys
from simple_cbow import SimpleCBOW
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from common.trainers import Trainer
from common.optimizers import Adam
from common.utils import process, create_contexts_targets, convert_one_hot
from common.log_config import logger

# 超参数
windows_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000


text = "You say goodbye and i say hello ."
corpus, word_to_id, id_to_word = process(text)

vocab_size = len(word_to_id)  # 单词个数
contexts, target = create_contexts_targets(corpus, windows_size)  # 生成上下文，目标词
logger.info(f"contexts: {contexts}")
logger.info(f"target: {target}")

target = convert_one_hot(target, vocab_size)
logger.info(f"one hot target: {target}")
contexts = convert_one_hot(contexts, vocab_size)
logger.info(f"one hot contexts : {contexts}")

model = SimpleCBOW(vocab_size, hidden_size)
logger.info(f"model layer_in0 param: {model.layer_in0.params}")
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()



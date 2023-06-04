# torch相关
import torch

# 图像
import numpy as np
import matplotlib.pyplot as plt

from config import logger
from train import train, model, figure_test, accu


train_core1 = {
    # 输入大小，因各图片尺寸不一，最好能保持一致
    "input_size": [3, 224, 224],
    # 分类数
    "class_dim": 12,
    # 学习率
    "lr": 0.0002,
    # 使用GPU
    "use_gpu": True,
    # 前期的训练轮数
    "num_epochs": 5,
    # 当达到想要的准确率就立刻保存下来当时的模型
    "last_acc": 0.4
}

# 记录此次运行的超参,方便日后做记录进行比对
logger.info(train_core1)

# train_ratio = 0.7
# train = open('data/train_split_list.txt', 'w')
# val = open('data/val_split_list.txt', 'w')
#
# with open("data/train_list.txt") as f:
#     for line in f.readlines():
#         if random.uniform(0, 1) <= train_ratio:
#             train.write(line)
#         else:
#             val.write(line)
# train.close()
# val.close()


'''
[[[0.485]],

 [[0.456]],

 [[0.406]]]
'''

if __name__ == '__main__':
    epochs = 10
    for epoch in range(epochs):
        train(epoch)

    model_name = "round %d.pkl" % epochs

    # 保存模型
    torch.save(model, model_name)

    figure_test()

    # 可视化
    accu = [np.array(x.cpu()) for x in accu]
    plt.plot(np.array(list(range(10))), accu)
    plt.show()

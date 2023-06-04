import numpy as np
import matplotlib.pyplot as plt

from two_layer_net import TwoLayerNet, SGD, load_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


W1 = np.random.randn(2, 4)  # 输入层权重
b1 = np.random.randn(4)  # 输入层偏置
W2 = np.random.randn(4, 3) # 输出层权重
b2 = np.random.randn(3)  # 输出层偏置
x = np.random.randn(10, 2)  # 输入

y = np.dot(x, W1) + b1
a = sigmoid(y)
s = np.dot(a, W2) + b2
# print(s)


# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 读入数组，生成模型和优化器
x, t = load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, out_put_size=3)
optimizer = SGD(learning_rate)

# 学习用到的数据
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 打乱数据
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters * batch_size:(iters+1)*batch_size]
        batch_t = t[iters * batch_size:(iters+1)*batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期输出结果
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f"| epoch {epoch + 1} | iters {iters + 1} / {max_iters} | loss %.2f" % avg_loss)
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0


# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

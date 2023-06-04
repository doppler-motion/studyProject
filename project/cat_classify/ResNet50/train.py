import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from data_loader import myData, load_data
from model import myRes

# dataset
img_datasets = {x: myData(x) for x in ['train', 'val']}
dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
# test_datasets = {'test': myData('test')}
# test_size = {'test': len(test_datasets)}
train_dataloader = DataLoader(
    dataset=img_datasets["train"],
    batch_size=10,
    shuffle=True
)
val_dataloader = DataLoader(
    dataset=img_datasets["val"],
    batch_size=1,
    shuffle=False
)

# 训练集
trainset = load_data(mode='train', shuffle=True, color_jitter=True, rotate=True)

# 验证集
valset = load_data(mode='val', shuffle=False, color_jitter=False, rotate=False)
# 测试集
# testset = load_data(mode='test', shuffle=False, color_jitter=False, rotate=False)


# 判断cuda是否可用
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = torch.device('cuda')
    model = myRes().to(device)
else:
    model = myRes()

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.5)

accu = []


def train(epoch):
    model.train()
    correct = 0
    sum_loss = 0.0
    for (data, target) in train_dataloader:
        data, target = Variable(data).cuda(), Variable(target).cuda()

        optim.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optim.step()

        _, id = torch.max(pred.data, 1)
        sum_loss += loss.data
        correct += torch.sum(id == target.data)

        # 尝试释放显存
        data = data.cpu()
        target = target.cpu()

        torch.cuda.empty_cache()
        # 变为cpu数据后并不会丢失
        # print(data.data)
        # print(correct.data)

    print('[epoch %d] loss:%.03f' % (epoch + 1, sum_loss.data / len(train_dataloader)))
    print('        correct:%.03f%%' % (100 * correct.data / len(trainset)))
    accu.append((correct.data / len(trainset)).data)


def figure_test():
    model.eval()
    with torch.no_grad():
        correct = 0
        for (data, target) in val_dataloader:
            data, target = Variable(data).cuda(), Variable(target).cuda()

            pred = model(data)

            _, id = torch.max(pred.data, 1)
            correct += torch.sum(id == target.data)
        print("test accu: %.03f%%" % (100 * correct / len(valset)))

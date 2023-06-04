# 1. 导入必要的库
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

from PIL import Image

# 2. 图片转换、增强策略
transform = {
    "train": transforms.Compose([
        transforms.Resize(300),
        transforms.RandomResizedCrop(300),  # 随机裁剪到300 * 300
        transforms.RandomHorizontalFlip(),  # 随机水平变换
        transforms.CenterCrop(256),  # 中心裁剪
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正则化
    ]),
    "val": transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),  # 中心裁剪
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 正则化
    ]),
    "test": transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 判断cuda是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3. 划分数据集
train_ratio = 0.8
train = open('dataset/cat_12/train_split_list.txt', 'w')
val = open('dataset/cat_12/val_split_list.txt', 'w')

with open("dataset/cat_12/train_list.txt") as f:
    for line in f.readlines():
        if np.random.uniform(0, 1) <= train_ratio:
            train.write(line)
        else:
            val.write(line)
train.close()
val.close()


# 4. 加载数据集
def image_loader(path):
    return Image.open(path).convert("RGB")


class MyDataset(Dataset):
    def __init__(self, mode, transform):
        self.mode = mode
        self.transform = transform

        self.imgs = self.load_data()

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            img_data = self.imgs[index]
            img, label = img_data[0], img_data[1]
            #             label = torch.from_numpy(label)
            img_path = f"./dataset/cat_12/cat_12_train/" + img
            img = image_loader(img_path)
            img = self.transform(img)

            return img, torch.tensor(int(label))

    def __len__(self):
        return len(self.imgs)

    def load_data(self):
        split_data_txt = "./dataset/cat_12/%s_split_list.txt" % self.mode

        if self.mode in ("train", "val"):
            data_df = pd.read_csv(split_data_txt, header=None, sep="\t")
            data_df.columns = ["path", "label"]
            data_df["path"] = data_df["path"].apply(lambda x: str(x).strip().split("/")[-1])
            data_df["label"] = data_df["label"].apply(lambda x: "%02d" % int(str(x).strip()))

            imgs = list(data_df[["path", "label"]].itertuples(index=False, name=None))

            return imgs


data_set = {x: MyDataset(mode=x, transform=transform[x]) for x in ("train", "val")}
# print(data_set["train"][0])

# 数据集大小
dataset_sizes = {x: len(data_set[x]) for x in ['train', 'val']}
# print(dataset_sizes)

data_loader = {"train": DataLoader(data_set["train"], batch_size=10, shuffle=True),
               "val": DataLoader(data_set["val"])}


# for data, target in data_loader["val"]:
#     print(data)
#     print(target)
#     break

# 5. 设计模型1
class MyRes(nn.Module):
    def __init__(self):
        super(MyRes, self).__init__()

        self.resnet = models.resnet50(pretrained=False)

    def forward(self, x):
        x = self.resnet(x)
        return x


# 设计模型2： 迁移学习
def get_model():
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 12)
    model_ft = model_ft.to(device)
    return model_ft


#  构建模型2
model2 = get_model()

criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.5)  # 优化器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率优化


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):  # 括号中的参数是模型，损失函数标准，优化器，学习速率衰减方式，训练次数

    best_model_wts = copy.deepcopy(model.state_dict())  # 先深拷贝一份当前模型的参数（wts=weights），后面迭代过程中若遇到更优模型则替换
    best_acc = 0.0  # 最佳正确率，用于判断是否替换best_model_wts

    for epoch in range(num_epochs):  # 开启每一个epoch
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))

        for phase in ['train', 'val']:  # 每个epoch中都包含训练与验证两个阶段
            if phase == 'train':
                model.train()
            else:
                model.eval()
                # 与train不同的是，test过程中没有batch-normalization与dropout，因此要区别对待。
                # batchnorm是针对minibatch的，测试过程中每个样本单独测试，不存在minibatch

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(
                        phase == 'train'):  # torch.set_grad_enabled(False/True)是上下文管理器，用于确定是否对with下的所有语句设计的参数求导，如果设置为false则新节点不可求导
                    outputs = model(inputs)  # 网络模型的前向传播，就是为了从输入得到输出
                    _, preds = torch.max(outputs, 1)  # 在维度1（行方向）查找最大值
                    loss = criterion(outputs, labels)  # 输出结果与label相比较

                    if phase == 'train':
                        loss.backward()  # 误差反向传播，计算每个w与b的更新值
                        optimizer.step()  # 将这些更新值施加到模型上

                # train, val都一样
                running_loss += loss.item() * inputs.size(0)  # 计算当前epoch过程中，所有batch的损失和
                running_corrects += torch.sum(preds == labels.data)  # 判断正确的样本数
            if phase == 'train':  # 完成本次epoch所有样本的训练与验证之后，就对学习速率进行修正
                scheduler.step()  # 在训练过程中，要根据损失的情况修正学习速率

            epoch_loss = running_loss / dataset_sizes[phase]  # 当前epoch的损失值是loss总和除以样本数
            epoch_acc = running_corrects.double() / dataset_sizes[phase]  # 当前epoch的正确率

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(  # 输出train/test，损失、正确率
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:  # 如果是val阶段，并且当前epoch的acc比best acc大
                best_acc = epoch_acc  # 就替换best acc为当前epoch的acc
                best_model_wts = copy.deepcopy(model.state_dict())  # 将best_model替换为当前模型

    print('Best val Acc: {:4f}'.format(best_acc))

    # 将最佳模型的相关参数加载到model中
    model.load_state_dict(best_model_wts)
    return model


# 6. 模型2训练
model_ft = train_model(model2, criterion, optimizer, exp_lr_scheduler, num_epochs=15)  # 模型训练


# 7. 模型2验证
def vali(M, dataset):
    M.eval()
    with torch.no_grad():
        correct = 0
        for (data, target) in data_loader['val']:
            data, target = data.to(device), target.to(device)

            pred = M(data)
            _, id = torch.max(pred, 1)
            correct += torch.sum(id == target.data)
        print("test accu: %.03f%%" % (100 * correct / len(dataset)))
    return (100 * correct / len(dataset)).item()


test_accu = int(vali(model_ft, data_set['val']) * 100)
print(test_accu)

# 构建模型1
# model1 = MyRes().cuda()
#
# criterion = nn.CrossEntropyLoss()  # 损失函数
# optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.5)  # 优化器
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率优化
# accu = []
#
#
# # 6. 模型1训练
# def train(epoch):
#     model1.train()
#     correct = 0
#     sum_loss = 0.0
#     for (data, target) in data_loader["train"]:
#         #         data, target = Variable(data).cuda(), Variable(target).cuda()
#         data, target = data.to(device), target.to(device)
#
#         optimizer.zero_grad()
#         pred = model1(data)
#         loss = criterion(pred, target)
#         loss.backward()
#         optimizer.step()
#
#         _, id = torch.max(pred.data, 1)
#         sum_loss += loss.data
#         correct += torch.sum(id == target.data)
#
#         # 尝试释放显存
#         data = data.cpu()
#         target = target.cpu()
#
#         torch.cuda.empty_cache()
#         # 变为cpu数据后并不会丢失
#         # print(data.data)
#         # print(correct.data)
#
#     print('[epoch %d] loss:%.03f' % (epoch + 1, sum_loss.data / len(data_set["train"])))
#     print('        correct:%.03f%%' % (100 * correct.data / len(data_set["train"])))
#     accu.append((correct.data / len(data_set["train"])).data)
#
#
# for epoch in range(15):
#     train(epoch)
# print(accu)
#
# # 7. 模型1验证
# model1.eval()
# with torch.no_grad():
#     correct = 0.0
#     # 迭代测试集
#     for images, labels in data_loader["val"]:
#         data, target = Variable(images).cuda(), Variable(labels).cuda()
#
#         pred = model1(data)
#
#         _, idx = torch.max(pred.data, 1)
#         correct += torch.sum(idx == target.data)
#     print("val accu: %.03f%%" % (100 * correct / (dataset_sizes["train"] + dataset_sizes["val"])))

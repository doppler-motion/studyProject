# 1. 加载数据
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 将tensor转换为image
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)

# 3 定义超参数
batch_size = 16
model_path = 'model.pth'
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# 4 图片转换,增强策略
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

# 5操作数据集
# 5.1 数据集路径
data_path = "./data"
# 5.2 加载数据集
image_dataset = {x: datasets.ImageFolder(os.path.join(data_path, x), transform[x])
                 for x in ("train", "val")}
# 5.3 创建迭代器, 读取数据
data_loaders = {x: DataLoader(image_dataset[x], shuffle=True, batch_size=batch_size) for x in ("train", "val")}

# 5.4 训练集和验证集的大小
data_sizes = {x: len(image_dataset[x]) for x in ("train", "val")}

# 5.5 获取标签的类别名称： NORMAL 正常 --- PNEUMONIA  感染
target_names = image_dataset["train"].classes
LABEL = dict((v, k) for k, v in image_dataset['train'].class_to_idx.items())


def misclassified_images(pred, writer, target, data, output, epoch, count=10):
    misclassified = (pred != target.data)  # 记录预测值与真实值不同的True和False
    for index, image_tensor in enumerate(data[misclassified][:count]):
        # 显示预测不同的前10张图片
        img_name = '{}->Predict-{}x{}-Actual'.format(
            epoch,
            LABEL[pred[misclassified].tolist()[index]],
            LABEL[target.data[misclassified].tolist()[index]],
        )
        writer.add_image(img_name, inv_normalize(image_tensor), epoch)


#  2 可视化图片
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# 8 池化层替换
class AdaptiveConcatPoll2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPoll2d, self).__init__()
        size = size or (1, 1)  # 池化层大小，默认为1
        self.pool1 = nn.AdaptiveAvgPool2d(size)  # 池化层1, 平均池化
        self.pool2 = nn.AdaptiveAvgPool2d(size)  # 池化层2, 最大值池化

    def forward(self, x):
        return torch.cat([self.pool1(x), self.pool2(x)], 1)  # 连接两个池化层


# 7 迁移学习：拿到一个成熟的模型，进行模型微调
def get_model():
    model_pre = models.resnet50(pretrained=True)  # 获取预训练模型
    # 冻结预训练模型中所有参数
    for param in model_pre.parameters():
        param.requires_grad = False
    # 模型微调：替换ResNet最后的两层网络，返回一个新的模型
    model_pre.avgpool = AdaptiveConcatPoll2d()  # 池化层替换
    model_pre.fc = nn.Sequential(
        nn.Flatten(),  # 所有维度拉平
        nn.BatchNorm1d(4096),  # 256 * 6 * 6  4096 36864
        nn.Dropout(0.5),  # 随机丢掉一些神经元
        nn.Linear(4096, 512),  # 线性层的处理
        nn.ReLU(),  # 激活层
        nn.BatchNorm1d(512),  # 正则化处理
        nn.Linear(512, 2),  #
        nn.LogSoftmax(dim=1),  # 损失函数
    )

    return model_pre


# 9 定义训练函数
def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    """
    作用：声明在模型训练时，采用Batch Normalization 和 Dropout
    Batch Normalization : 对网络中间的每层进行归一化处理，保证每层所提取的特征分布不会被破坏
    Dropout : 减少过拟合
    Parameters
    ----------
    model
    device
    train_loader
    criterion
    optimizer
    epoch
    writer

    Returns
    -------

    """
    model.train()
    total_loss = 0.0  # 总损失值
    # 循环读取训练数据，更新模型参数
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 梯度初始化为0
        output = model(data)  # 训练后的输出
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累加总的损失值

    writer.add_scalar("Train Loss", total_loss / len(train_loader), epoch)
    writer.flush()  # 刷新
    return total_loss / len(train_loader)  # 返回平均损失值


# 10 定义测试函数
def test(model, device, test_loader, criterion, epoch, writer):
    """
    # 作用：声明在模型训练时，不采用Batch Normalization 和 Dropout
    Parameters
    ----------
    model
    device
    test_loader
    criterion
    epoch
    writer

    Returns
    -------

    """
    model.eval()
    # 损失和正确率
    total_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 预测输出
            output = model(data)
            # 计算损失
            total_loss += criterion(output, target).item()
            # 获取预测结果中每行数据概率最大的下标
            _, preds = torch.max(output, dim=1)
            # 累计预测正确的个数
            correct += torch.sum(preds == target.data)

            ######## 增加 #######
            misclassified_images(preds, writer, target, data, output, epoch)  # 记录错误分类的图片

        # 平均损失
        total_loss /= len(test_loader)
        # 正确率
        accuracy = correct / len(test_loader.dataset)
        # 写入日志
        writer.add_scalar("Test Loss", total_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        # 刷新
        writer.flush()
        # 输出信息
        print("Test Loss: {:.4f}, Accuracy: {:.4f}".format(total_loss, accuracy))
        return total_loss, accuracy


# 定义函数，获取Tensorboard的writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter('logdir/' + timestr)
    return writer


def train_epochs(model, device, dataloaders, criterion, optimizer, num_epochs, writer):
    """
    Returns:
        返回一个训练过后最好的模型
    """
    print("{0:>20} | {1:>20} | {2:>20} | {3:>20} |".format('Epoch', 'Training Loss', 'Test Loss', 'Accuracy'))
    best_score = np.inf  # 假设最好的预测值
    start = time.time()  # 开始时间

    # 开始循环读取数据进行训练和验证
    for epoch in num_epochs:

        train_loss = train(model, device, dataloaders['train'], criterion, optimizer, epoch, writer)

        test_loss, accuracy = test(model, device, dataloaders['val'], criterion, epoch, writer)

        if test_loss < best_score:
            best_score = test_loss
            torch.save(model.state_dict(), model_path)  # 保存模型 # state_dict变量存放训练过程中需要学习的权重和偏置系数

        print("{0:>20} | {1:>20} | {2:>20} | {3:>20.2f} |".format(epoch, train_loss, test_loss, accuracy))

        writer.flush()

    # 训练完所耗费的总时间
    time_all = time.time() - start
    # 输出时间信息
    print("Training complete in {:.2f}m {:.2f}s".format(time_all // 60, time_all % 60))


def main():
    # 7 显示batch_size的图片
    # 读取 8张图片
    datas, targets = next(iter(data_loaders["train"]))
    # 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 显示图片
    image_show(out, title=[target_names[x] for x in targets])

    # 定义writer
    writer = tb_writer()
    images, labels = next(iter(data_loaders["train"]))  # 获取一批数据
    grid = make_grid([inv_normalize(image) for image in images[:32]])  # 读取32张图片
    writer.add_image("X-ray grid", grid, 0)  # 添加到tensorboard
    writer.flush()

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(model.parameters())
    train_epochs(model, device, data_loaders, criterion, optimizer, range(0, 10), writer)
    writer.close()


if __name__ == "__main__":
    main()

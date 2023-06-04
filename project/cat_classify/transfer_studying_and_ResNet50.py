# Dataset
import os
import cv2
import torch
import random
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import numpy as np
import math
import copy
from PIL import Image, ImageEnhance

batches = 10
DATA_DIM = 224

THREAD = 8
BUF_SIZE = 102400

# 为什么0-255的像素值的mean和std在(0, 1)---->见157行
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

train_ratio = 0.7
train = open('data_process/train_split_list.txt', 'w')
val = open('data_process/val_split_list.txt', 'w')

with open("data_process/train_list.txt") as f:
    for line in f.readlines():
        if random.uniform(0, 1) <= train_ratio:
            train.write(line)
        else:
            val.write(line)
train.close()
val.close()


# 简单放缩
def resize_short(img, target_size):
    '''
    根据输入的img和target_size, 返回经过多相位图象插值算法处理过后的放大或缩小的图片
    '''
    percent = float(target_size) / min(img.size[0], img.size[1])
    resized_width = int(round(img.size[0] * percent))
    resized_height = int(round(img.size[1] * percent))
    img = img.resize((resized_width, resized_height), Image.LANCZOS)  # LANCZOS 多相位图象插值算法
    return img


# 裁剪
def crop_image(img, target_size, center):
    '''
    center表示中心，否则随机裁剪
    target表示裁剪后的尺寸
    '''
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img


# 随机裁剪
def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    '''
    通过一系列骚操作，确定随即裁剪的起始点和裁剪长和裁剪宽
    返回裁剪区域经过resize的统一尺寸图像
    '''
    aspect_ratio = math.sqrt(
        np.random.uniform(*ratio)  # 在ratio = [0.75, 1.333]之间随机生成浮点数。*是解引用、等同于(ratio[0], ratio[1])
    )  # 0.86 -- 1.15
    w = 1.0 * aspect_ratio
    h = 1.0 / aspect_ratio

    bound = min(
        float(img.size[0] / img.size[1]) / (w ** 2),
        float(img.size[1] / img.size[0]) / (h ** 2),
    )
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    # 确定裁剪区域
    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min, scale_max)

    # 确定裁剪大小
    target_size = math.sqrt(target_area)
    w = int(target_size * w)  # 裁剪的宽
    h = int(target_size * h)  # 裁剪的高

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.LANCZOS)
    return img


# 旋转角度
def rotate_image(img):
    angle = np.random.randint(-10, 11)
    img = img.rotate(angle)
    return img


# 颜色增强
def distort_color(img):
    '''
    数据增强

    随机改变图片颜色的参数，如曝光度，对比度，颜色
    '''

    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)  # 增强幅度
        return ImageEnhance.Brightness(img).enhance(e)

    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)  # 增强幅度
        return ImageEnhance.Contrast(img).enhance(e)

    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)  # 增强幅度
        return ImageEnhance.Color(img).enhance(e)

    # 随机选择一种增强顺序
    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)

    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)

    return img


# 处理图片
def process_image(sample, mode, color_jitter, rotate):
    '''
    : params
    sample: (图片地址, label)
    mode: train or val or test
    color_jitter: 颜色增强(0 or 1)
    rotate: 0 or 1

    : return
    train和val模式返回: img, label
    test: img
    '''
    img_path = sample[0]
    img = Image.open(img_path)

    # 图像增强
    if mode == 'train':
        if rotate:
            img = rotate_image(img)
        img = random_crop(img, DATA_DIM)
    else:
        # val and test
        img = resize_short(img, target_size=256)
        img = crop_image(img, target_size=DATA_DIM, center=True)

    if mode == 'train':
        if color_jitter:
            img = distort_color(img)
        if np.random.randint(0, 2) == 1:
            # 以百分之五十的概率镜像图片
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # 非三通道转换为三通道
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 将图像从(w, h, c) 变为 (c, w, h)，并缩放到0 1区间
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255
    # 0，1区间内像素值标准化
    img -= img_mean
    img /= img_std

    if mode == 'train' or mode == 'val':
        return img, int(sample[1])
    elif mode == 'test':
        return [img]


class myData(Dataset):
    def __init__(self, kind):
        super(myData, self).__init__()
        self.mode = kind
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        if kind == 'test':
            self.imgs = self.load_origin_data()

        elif kind == 'train':
            self.imgs, self.labels = self.load_origin_data()
            # self.imgs, self.labels = self.enlarge_dataset(kind, self.imgs, self.labels, 0.5)
            print('train size:')
            print(len(self.imgs))

        else:
            self.imgs, self.labels = self.load_origin_data()

    def __getitem__(self, index):
        if self.mode == 'test':
            sample = self.transform(self.imgs[index])
            return sample
        else:
            sample = self.transform(self.imgs[index])
            return sample, torch.tensor(self.labels[index])

    def __len__(self):
        return len(self.imgs)

    def load_origin_data(self):
        filelist = './data/%s_split_list.txt' % self.mode

        imgs = []
        labels = []
        data_dir = os.getcwd()
        if self.mode == 'train' or self.mode == 'val':
            with open(filelist) as flist:
                lines = [line.strip() for line in flist]
                if self.mode == 'train':
                    np.random.shuffle(lines)
                for line in lines:
                    # print(f"line: {line}")
                    img_path, label = line.split('\t')
                    img_path = os.path.join(data_dir + "/data/", img_path)
                    # print(f"img_path: {img_path}")
                    try:
                        # img, label = process_image((img_path, label), mode, color_jitter, rotate)
                        img = Image.fromarray(cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1))
                        imgs.append(img)
                        labels.append(int(label))
                    except:
                        print(img_path)
                        continue
                return imgs, labels
        elif self.mode == 'test':
            full_lines = os.listdir('data_process/cat_12_test/')
            lines = [line.strip() for line in full_lines]
            for img_path in lines:
                img_path = os.path.join(data_dir + "/data/cat_12_test/", img_path)
                # try:
                #     img= process_image((img_path, label), mode, color_jitter, rotate)
                #     imgs.append(img)
                # except:
                #     print(img_path)
                # img = Image.open(img_path)
                try:
                    img = Image.fromarray(cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1))
                    imgs.append(img)
                except:
                    print("img_path: ", img_path)
            return imgs

    def load_data(self, mode, shuffle, color_jitter, rotate):
        '''
        :return : img, label
        img: (channel, w, h)
        '''
        filelist = './data/%s_split_list.txt' % mode

        imgs = []
        labels = []
        data_dir = os.getcwd()
        if mode == 'train' or mode == 'val':
            with open(filelist) as flist:
                lines = [line.strip() for line in flist]
                if shuffle:
                    np.random.shuffle(lines)

                for line in lines:
                    img_path, label = line.split('&')
                    img_path = os.path.join(data_dir + "/data/", img_path)
                    try:
                        img, label = process_image((img_path, label), mode, color_jitter, rotate)
                        imgs.append(img)
                        labels.append(label)
                    except:
                        # print(img_path)
                        continue
                return imgs, labels

        elif mode == 'test':
            full_lines = os.listdir('data_process/test/')
            lines = [line.strip() for line in full_lines]
            for img_path in lines:
                img_path = os.path.join(data_dir, "data_process/cat_12_test/", img_path)
                # try:
                #     img= process_image((img_path, label), mode, color_jitter, rotate)
                #     imgs.append(img)
                # except:
                #     print(img_path)
                img = process_image((img_path, 0), mode, color_jitter, rotate)
                imgs.append(img)
            return imgs


# dataset
# img_datasets = {x: myData(x) for x in ['train', 'val']}
# dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val']}
# test_datasets = {'test': myData('test')}
# test_size = {'test': len(test_datasets)}


img_datasets = {x: myData(x) for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(img_datasets[x]) for x in ['train', 'val', 'test']}

# dataset准备完毕，开始创建dataloader
train_loader = DataLoader(
    dataset=img_datasets['train'],
    batch_size=batches,
    shuffle=True
)

val_loader = DataLoader(
    dataset=img_datasets['val'],
    batch_size=1,
    shuffle=False
)

test_loader = DataLoader(
    dataset=img_datasets['test'],
    batch_size=1,
    shuffle=False
)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}

# train
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

            for inputs, labels in dataloaders[phase]:
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


# 迁移学习
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 12)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)  # 模型训练


def vali(M, dataset):
    M.eval()
    with torch.no_grad():
        correct = 0
        for (data, target) in val_loader:
            data, target = data.to(device), target.to(device)

            pred = M(data)
            _, id = torch.max(pred, 1)
            correct += torch.sum(id == target.data)
        print("test accu: %.03f%%" % (100 * correct / len(dataset)))
    return (100 * correct / len(dataset)).item()


test_accu = int(vali(model_ft, img_datasets['val']) * 100)

model_name = 'val_{}.pkl'.format(test_accu)

torch.save(model_ft.state_dict(), os.path.join("./myModels", model_name))

# 加载模型
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 12)  # 注意更改维度
model_ft = model_ft.to(device)  # 注意要放入gpu，保持和参数数据类型一致

model_ft.load_state_dict(torch.load(f"./myModels/{model_name}"))
vali(model_ft, img_datasets['val'])

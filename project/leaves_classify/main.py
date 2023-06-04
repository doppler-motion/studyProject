# 首先导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

# 看一下是在cpu还是GPU上
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 超参数
learning_rate = 3e-4
weight_decay = 1e-3
num_epoch = 15
batch_size = 32
model_path = '../model/pre-resnext-model/pre_resnext_model .ckpt'

# 1. 读取数据 看看label文件长啥样
path = './data/classify-leaves'
labels_file_path = os.path.join(path, 'train.csv')
sample_submission_path = os.path.join(path, 'test.csv')

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
labels_unique = df['label'].unique()

leaves_labels = sorted(list(set(df['label'])))
n_classes = len(leaves_labels)
# 把label转成对应的数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))


# 2. 数据增强
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1), ratio=(0.8, 1.2)),  # 随机剪裁
        transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 颜色亮度色调
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def get_valid_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# 3. 创建dataset
class LeafDataset(Dataset):
    def __init__(self, images_filepaths, labels, transform=None):
        self.image_paths = np.asarray(images_filepaths)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = Image.open(image_filepath)
        # image = cv2.imread(image_filepath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        num_label = class_to_num[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, num_label


# 4. 构建模型
# 是否要冻住模型的前面一些层
def set_parameter_requires_grad(model, feature_extracting):
    #     if feature_extracting:
    #         model = model
    #         for i, param in enumerate(model.children()):
    #             if i == 8:
    #                 break
    #             param.requires_grad = False
    if feature_extracting:
        model = model
        for param in model.parameters():
            param.requires_grad = False


# resnet34模型
def res_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    #     model_ft.fc = nn.Sequential(
    #         nn.Linear(num_ftrs, 512),
    #         nn.ReLU(inplace=True),
    #         nn.Dropout(.3),
    #         nn.Linear(512, len(num_to_class))
    #     )
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )
    return model_ft


# resnext50模型
def resnext_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft


# Initialize a model, and put it on the device specified.
# model = res_model(176)
model = resnext_model(176)
model = model.to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

# The number of training epochs.
n_epochs = num_epoch
best_acc = 0.0

kf = StratifiedKFold(n_splits=5)

for k, (train_index, test_index) in enumerate(kf.split(df['image'], df['label'])):
    train_img, valid_img = df['image'][train_index], df['image'][test_index]
    train_labels, valid_labels = df['label'][train_index], df['label'][test_index]

    train_paths = './data/classify-leaves/' + train_img
    valid_paths = './data/classify-leaves/' + valid_img
    test_paths = './data/classify-leaves/' + sub_df['image']

    train_dataset = LeafDataset(images_filepaths=train_paths.values,
                                labels=train_labels.values,
                                transform=get_train_transform())
    valid_dataset = LeafDataset(images_filepaths=valid_paths.values,
                                labels=valid_labels.values,
                                transform=get_valid_transform())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False
    )

    for epoch in range(n_epochs):
        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        # These are used to record information in training.
        train_loss = []
        train_accs = []
        i = 0
        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.

            # print(batch)
            imgs, labs = batch

            imgs = imgs.to(device)
            labs = labs.to(device)
            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs)
            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labs)

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()
            # Compute the gradients for parameters.
            loss.backward()
            # Update the parameters with computed gradients.
            optimizer.step()
            # 更新学习率------------------------------------------------------------------------------
            scheduler.step()
            if (i % 500 == 0):
                print("learning_rate:", scheduler.get_last_lr()[0])
            i = i + 1

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labs).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()
        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(val_loader):
            imgs, labs = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labs.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labs.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # if the model improves, save a checkpoint at this epoch
        if valid_acc > best_acc:
            best_acc = valid_acc
            print("Epoch: {epoch}. Train.      {best_acc}".format(epoch=epoch, best_acc=best_acc))
            if best_acc > 0.80:
                torch.save(model.state_dict(),
                           f"./model/resnext50_32x4d_{k}flod_{epoch}epochs_accuracy{valid_acc:.5f}_weights.pth")
                print('saving model with acc {:.3f}'.format(best_acc))

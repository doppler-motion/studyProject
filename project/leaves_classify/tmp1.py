import os
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

# 1. 读取数据 看看label文件长啥样
path = './data/classify-leaves'
labels_file_path = os.path.join(path, 'train.csv')
sample_submission_path = os.path.join(path, 'test.csv')

df = pd.read_csv(labels_file_path)


# 2. 数据增强
def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1), ratio=(0.8, 1.2)),  # 随机剪裁
        transforms.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # 颜色亮度色调
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.RandomVerticalFlip(p=0.5),  # 随机水平翻转 选择一个概率
        transforms.ToTensor()
    ])


def get_valid_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
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
        if self.transform is not None:
            image = self.transform(image)
        return image, label


kf = StratifiedKFold(n_splits=5)

for k, (train_index, test_index) in enumerate(kf.split(df['image'], df['label'])):
    train_img, valid_img = df['image'][train_index], df['image'][test_index]
    train_labels, valid_labels = df['label'][train_index], df['label'][test_index]

    train_paths = './data/classify-leaves/' + train_img
    valid_paths = './data/classify-leaves/' + valid_img

    train_dataset = LeafDataset(images_filepaths=train_paths.values,
                                labels=train_labels.values,
                                transform=get_train_transform())
    valid_dataset = LeafDataset(images_filepaths=valid_paths.values,
                                labels=valid_labels.values,
                                transform=get_valid_transform())
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    val_loader = DataLoader(
        valid_dataset, batch_size=16, shuffle=False
    )
    for batch in train_loader:
        imgs, labels = batch
        #     print(batch)
        print(type(labels))
        break

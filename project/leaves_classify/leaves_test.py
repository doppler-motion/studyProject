import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 看一下是在cpu还是GPU上
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# 读取数据
path = './data/classify-leaves'
labels_file_path = os.path.join(path, 'train.csv')
sample_submission_path = os.path.join(path, 'test.csv')

df = pd.read_csv(labels_file_path)
sub_df = pd.read_csv(sample_submission_path)
labels_unique = df['label'].unique()

le = LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
label_inv_map = {v: k for k, v in label_map.items()}

train_img, valid_img = df['image'], df['image']
train_labels, valid_labels = df['label'], df['label']

train_paths = './data/classify-leaves/' + train_img
valid_paths = './data/classify-leaves/' + valid_img
test_paths = './data/classify-leaves/' + sub_df['image']

# model_name = ['seresnext50_32x4d', 'resnet50d']
model_path_list = [
    './model/resnext50_32x4d_0flod_12epochs_accuracy0.87066_weights.pth',
    './model/resnext50_32x4d_1flod_5epochs_accuracy0.93848_weights.pth',
    './model/resnext50_32x4d_2flod_8epochs_accuracy0.94565_weights.pth',
    './model/resnext50_32x4d_3flod_6epochs_accuracy0.95679_weights.pth',
    './model/resnext50_32x4d_4flod_2epochs_accuracy0.97228_weights.pth',
]


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
        # num_label = class_to_num[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


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


# resnext50模型
def resnext_model(num_classes, feature_extract=False, use_pretrained=True):
    model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))

    return model_ft


def get_valid_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


model_list = []
for i in range(len(model_path_list)):
    if i < 5:
        model_list.append(resnext_model(176))
    model_list[i] = nn.DataParallel(model_list[i])
    model_list[i] = model_list[i].to(device)
    init = torch.load(model_path_list[i])
    model_list[i].load_state_dict(init, strict=False)
    model_list[i].eval()
    model_list[i].cuda()

labels = np.zeros(len(test_paths))  # Fake Labels
test_dataset = LeafDataset(images_filepaths=test_paths,
                           labels=labels,
                           transform=get_valid_transform())
test_loader = DataLoader(
    test_dataset, batch_size=128, shuffle=False,
    # num_workers=10, pin_memory=True
)

predicted_labels = []
pred_string = []
preds = []

with torch.no_grad():
    for (images, target) in test_loader:
        images = images.cuda()
        onehots = sum([model(images) for model in model_list]) / len(model_list)
        for oh, name in zip(onehots, target):
            lbs = label_inv_map[torch.argmax(oh).item()]
            preds.append(dict(image=name, labels=lbs))

df_preds = pd.DataFrame(preds)
sub_df['label'] = df_preds['labels']
sub_df.to_csv('submission.csv', index=False)
sub_df.head()

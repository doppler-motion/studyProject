import os
import numpy as np
import cv2
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from process_img import process_image


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
            print('train size:', len(self.imgs))

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
                print(f"train file num : {len(lines)}")
                if self.mode == 'train':
                    np.random.shuffle(lines)
                for line in lines:
                    img_path, label = line.split('\t')
                    img_path = os.path.join(data_dir + "/data/", img_path)
                    try:
                        # img, label = process_image((img_path, label), mode, color_jitter, rotate)
                        np_file = np.fromfile(img_path, dtype=np.uint8)
                        # print(f"np_file: {np_file}")
                        cv2_file = cv2.imdecode(np_file, 1)
                        # print(f"cv2_file: {cv2_file}")
                        img = Image.fromarray(cv2_file)
                        # img = Image.fromarray(cv2.imdecode(np.fromfile(img_path, dtype=np.float32), 1))
                        imgs.append(img)
                        labels.append(int(label))
                    except:
                        print(img_path)
                        break
                        # continue
                return imgs, labels
        elif self.mode == 'test':
            full_lines = os.listdir('cat_12_test/')
            lines = [line.strip() for line in full_lines]
            for img_path in lines:
                img_path = os.path.join(data_dir, "cat_12_test/", img_path)
                # try:
                #     img= process_image((img_path, label), mode, color_jitter, rotate)
                #     imgs.append(img)
                # except:
                #     print(img_path)
                # img = Image.open(img_path)
                try:
                    np_file = np.fromfile(img_path, dtype=np.uint8)
                    print(f"np_file: {np_file}")
                    cv2_file = cv2.imdecode(np_file, 1)
                    print(f"cv2_file: {cv2_file}")
                    img = Image.fromarray(cv2_file)
                    # img = Image.fromarray(cv2.imdecode(np.fromfile(img_path, dtype=np.float32), 1))
                    imgs.append(img)
                except:
                    print(img_path)
                    continue
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
            full_lines = os.listdir('cat_12_test/')
            lines = [line.strip() for line in full_lines]
            for img_path in lines:
                img_path = os.path.join(data_dir, "cat_12_test/", img_path)
                # try:
                #     img= process_image((img_path, label), mode, color_jitter, rotate)
                #     imgs.append(img)
                # except:
                #     print(img_path)
                img = process_image((img_path, 0), mode, color_jitter, rotate)
                imgs.append(img)
            return imgs


def load_data(mode='train', shuffle=False, color_jitter=False, rotate=False):
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
                img_path, label = line.split('\t')
                img_path = img_path.replace('JEPG', 'jepg')
                img_path = os.path.join(data_dir+"/data/", img_path)
                try:
                    img, label = process_image((img_path, label), mode, color_jitter, rotate)
                    imgs.append(img)
                    labels.append(label)
                except:
                    print(img_path)
                    continue
            return imgs, labels

    elif mode == 'test':
        full_lines = os.listdir('data/cat_12_test')
        lines = [line.strip() for line in full_lines]
        for img_path in lines:
            img_path = img_path.replace('JEPG', 'jepg')
            img_path = os.path.join(data_dir, "/data/cat_12_test/", img_path)
            # try:
            #     img= process_image((img_path, label), mode, color_jitter, rotate)
            #     imgs.append(img)
            # except:
            #     print(img_path)
            img = process_image((img_path, 0), mode, color_jitter, rotate)
            imgs.append(img)
        return imgs

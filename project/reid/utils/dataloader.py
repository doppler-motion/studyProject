from __future__ import print_function, absolute_import

import os

from PIL import Image
from torch.utils.data import Dataset

from .datamanager import init_img_dataset


def read_img(img_path):
    img = None
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))

    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, pid, camid = self.dataset[idx]
        img = read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


if __name__ == "__main__":
    dataset = init_img_dataset(name="market1501")
    train_loader = ImageDataset(dataset.train)
    for batch_id, (image, pid, camid) in enumerate(train_loader):
        print(batch_id)
        print(image)
        print(pid)
        print(camid)
        break

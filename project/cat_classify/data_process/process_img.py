import os
import cv2
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 102400

# 为什么0-255的像素值的mean和std在(0, 1)---->见157行
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
'''
[[[0.485]],

 [[0.456]],

 [[0.406]]]
'''


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


def load_data(mode, shuffle, color_jitter, rotate):
    '''
    :return : img, label
    img: (channel, w, h)
    '''
    filelist = '%s_split_list.txt' % mode

    imgs = []
    labels = []
    data_dir = os.getcwd()
    if mode == 'train' or mode == 'val':
        with open(filelist) as flist:
            lines = [line.strip() for line in flist]
            print(f"train file num: {len(lines)}")
            if shuffle:
                np.random.shuffle(lines)

            for line in lines:
                img_path, label = line.split('\t')
                img_path = os.path.join(data_dir, img_path)
                try:
                    print(f"img_path: {img_path}")
                    np_file = np.fromfile(img_path, dtype=np.float32)
                    print(f"np_file: {np_file}")
                    cv2_file = cv2.imdecode(np_file, 1)
                    print(f"cv2_file: {cv2_file}")
                    img = Image.fromarray(cv2_file)
                    # img, label = process_image((img_path, label), mode, color_jitter, rotate)
                    imgs.append(img)
                    labels.append(label)
                except:
                    print(img_path)
                    break
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
            np_file = np.fromfile(img_path, dtype=np.float32)
            print(f"np_file: {np_file}")
            cv2_file = cv2.imdecode(np_file, 1)
            print(f"cv2_file: {cv2_file}")
            img = Image.fromarray(cv2_file)
            # img = process_image((img_path, 0), mode, color_jitter, rotate)
            imgs.append(img)
        return imgs


# 将图片模式都转换为RGB模式
with open("train_split_list.txt", "r") as tfile:
    train_infos = tfile.readlines()

    train_infos = [file.strip() for file in train_infos]
    for train_file in train_infos:
        image_path, labels = train_file.split("\t")
        img = Image.open(image_path)
        if img.mode != "RGB":
            print("img path: ", image_path)
            print("img mode: ", img.mode)
            img = img.convert("RGB")
            img.save(image_path)

with open("val_split_list.txt", "r") as vfile:
    val_infos = vfile.readlines()

    val_infos = [file.strip() for file in val_infos]
    for val_file in val_infos:
        image_path, labels = val_file.split("\t")
        img = Image.open(image_path)
        if img.mode != "RGB":
            print("img path: ", image_path)
            print("img mode: ", img.mode)
            img = img.convert("RGB")
            img.save(image_path)

for image_path in os.listdir("cat_12_test/"):
    src = os.path.join("cat_12_test/", image_path)
    img = Image.open(src)
    if img.mode != "RGB":
        print(image_path)
        img = img.convert("RGB")
        img.save(src)

# 画图
img1 = Image.open("cat_12_train/loG9VQ4H1BNjEaDFRwyvxdcbrAWCTnph.jpg")
img2 = Image.open("cat_12_train/SANI5VsGngXkz3T6rWKYbjp9HO7ioBRm.jpg")
img3 = Image.open("cat_12_train/yGcJHV8Uuft6grFs7QWnK5CTAZvYzdDO.jpg")

print(f"img1 size: {img1.size}")
print(f"img1 mode: {img1.mode}")
print(f"img1 format: {img1.format}")
print(f"img2 size: {img2.size}")
print(f"img2 mode: {img2.mode}")
print(f"img2 format: {img2.format}")
print(f"img3 size: {img3.size}")
print(f"img3 mode: {img3.mode}")
print(f"img3 format: {img3.format}")

np_file1 = np.fromfile("cat_12_train/loG9VQ4H1BNjEaDFRwyvxdcbrAWCTnph.jpg", dtype=np.uint8)
np_file2 = np.fromfile("cat_12_train/SANI5VsGngXkz3T6rWKYbjp9HO7ioBRm.jpg", dtype=np.uint8)
np_file3 = np.fromfile("cat_12_train/yGcJHV8Uuft6grFs7QWnK5CTAZvYzdDO.jpg", dtype=np.uint8)
print("np_file1: ", np_file1)
print("np_file2: ", np_file2)
print("np_file3: ", np_file3)
print("np_file1 if have nan: ", np.isnan(np_file1).any())
print("np_file2 if have nan: ", np.isnan(np_file2).any())
print("np_file3 if have nan: ", np.isnan(np_file3).any())

index1 = np.argwhere(np.isnan(np_file1))
print(f"index1: {index1}")
index2 = np.argwhere(np.isnan(np_file2))
print(f"index2: {index2}")
index3 = np.argwhere(np.isnan(np_file3))
print(f"index3: {index3}")

cv2_file1 = cv2.imdecode(np_file1, 1)
cv2_file2 = cv2.imdecode(np_file2, 1)
cv2_file3 = cv2.imdecode(np_file3, 1)
print("cv2_file1: ", cv2_file1)
print("cv2_file2: ", cv2_file2)
print("cv2_file3: ", cv2_file3)

# plt.figure()
# ax1 = plt.subplot(2, 2, 1)
# plt.imshow(img1)
# ax2 = plt.subplot(2, 2, 2)
# plt.imshow(img2)
# plt.show()


# filelist = os.listdir("cat_12_test")
# print(filelist)
#
# for file in filelist:
#     print("file: ", file)
#     img = Image.open("cat_12_test/"+file)
#     print("img mode: ", img.mode)
#     # np_file = np.fromfile("cat_12_test/"+file, dtype=np.float32)
#     # print("np_file: ", np_file)
#     # cv2_file = cv2.imdecode(np_file, 1)
#     # print("cv2_file: ", cv2_file)
#     # img_file = Image.fromarray(cv2_file)
#     # print("img_file:", img_file)

import math
import numpy as np
import random

from PIL import Image, ImageEnhance

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 102400

# 为什么0-255的像素值的mean和std在(0, 1)---->见157行
img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))


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
    通过一系列操作，确定随即裁剪的起始点和裁剪长和裁剪宽
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

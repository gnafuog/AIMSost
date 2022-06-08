import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset


def data_augmentation():
    # 输入的数据增强，分别旋转90，180，270度
    directory_name = 'dataset/imgs'
    for filename in os.listdir(directory_name):
        img = Image.open(directory_name + '/' + filename)
        filename = filename[0:filename.find('.')]
        img2 = img.rotate(90)
        img3 = img.rotate(180)
        img4 = img.rotate(270)
        img5 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img6 = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save('dataset/data_x/' + filename + '.png')
        img2.save('dataset/data_x/' + filename + '_r90.png')
        img3.save('dataset/data_x/' + filename + '_r180.png')
        img4.save('dataset/data_x/' + filename + '_r270.png')
        img5.save('dataset/data_x/' + filename + '_flr.png')
        img6.save('dataset/data_x/' + filename + '_ftb.png')

    # 标签的数据增强，分别旋转90，180，270度
    directory_name = 'dataset/masks'
    for filename in os.listdir(directory_name):
        img = Image.open(directory_name + '/' + filename)
        filename = filename[0:filename.find('.')]
        img2 = img.rotate(90)
        img3 = img.rotate(180)
        img4 = img.rotate(270)
        img5 = img.transpose(Image.FLIP_LEFT_RIGHT)
        img6 = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save('dataset/data_y/' + filename + '.png')
        img2.save('dataset/data_y/' + filename + '_r90.png')
        img3.save('dataset/data_y/' + filename + '_r180.png')
        img4.save('dataset/data_y/' + filename + '_r270.png')
        img5.save('dataset/data_y/' + filename + '_flr.png')
        img6.save('dataset/data_y/' + filename + '_ftb.png')


def img_norm(a):
    t = []
    i_max = a.max()
    i_min = a.min()
    for i in a:
        i_norm = (i - i_min)/(i_max - i_min)
        t.append(i_norm)
    return t


def image_load():
    # data_augmentation()  # 如果要做数据增强就取消注释
    input_data = []
    label_data = []
    input_dir = 'dataset/data_x'
    label_dir = 'dataset/data_y'
    # np.set_printoptions(threshold=np.inf, suppress=True)
    # for filename in os.listdir(input_dir):
    #     img = Image.open(input_dir + '/' + filename)
    #     # 图像通道转换
    #     img = img.convert('L')
    #     # 图片像素归一化
    #     a = img_norm(np.array(img))
    #     input_data.append(a)
    # input_data = np.array(input_data)

    # for filename in os.listdir(label_dir):
    #     img = Image.open(label_dir + '/' + filename)
    #     img = img.convert('L')
    #     label_data.append(np.array(img))

    # 标签像素值归一化
    # label_data = np.minimum(label_data, 1)

    # 增添维数
    # input_data = np.expand_dims(input_data, axis=1)

    return input_data, label_data


def data_deal(batch_size: int, shuffle: bool = True, train_ratio=0.8, val_ratio=0.1, random: bool = True):
    # dataset_x, dataset_y = image_load()

    # input_dir = 'dataset/data_x/'
    input_dir = 'dataset/imgs/'
    label_dir = 'dataset/data_y/'
    positive = 0
    total = 0
    for filename in os.listdir(label_dir):
        img = Image.open(label_dir + filename)
        img = img.convert('L')
        y = np.minimum(np.array(img), 1)
        positive += y.sum()
        total += np.where(y == y, 1, 1).sum()

    negative_rate = 1 - positive / total

    dataset = ImageDataset(img_path=input_dir, mask_path=label_dir)
    dataset_len = len(dataset)
    train_size = int(train_ratio * dataset_len)
    val_size = int(val_ratio * dataset_len)
    test_size = dataset_len - train_size - val_size
    if random is True:
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), negative_rate, (1, 512, 512)
    if val_size == 0:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(test_dataset, batch_size=batch_size), negative_rate, (1, 512, 512)
    elif test_size == 0:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size), negative_rate, (1, 512, 512)
    else:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle), DataLoader(val_dataset, batch_size=batch_size), DataLoader(test_dataset, batch_size=batch_size), negative_rate, (1, 512, 512)


class ImageDataset(Dataset):

    def __init__(self, img_path, mask_path):
        self.img_path = img_path
        self.mask_path = mask_path
        self.files = os.listdir(self.img_path)

    def __getitem__(self, index):
        file = self.files[index]
        # 读取图片并转换为灰度格式
        img = Image.open(self.img_path + file).convert('L')

        # 做最值归一化
        img = img_norm(np.array(img))

        img = np.expand_dims(np.array(img), 0)
        # 读取mask
        mask = Image.open(self.mask_path + file).convert('L')
        # 将标签设置为0或1
        mask = np.minimum(np.array(mask), 1)
        return torch.FloatTensor(img), torch.FloatTensor(mask)

    def __len__(self):
        return len(self.files)

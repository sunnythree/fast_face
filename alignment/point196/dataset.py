import os
import torch
from torchvision import transforms as tfs
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import point196.config as cfg
from PIL import Image, ImageDraw, ImageFont
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def get_max_box(ann, size, need_random=True, rand_radio=0.3):
    min_x = 100000
    max_x = 0
    min_y = 100000
    max_y = 0
    for i in ann:
        if i[0] < min_x:
            min_x = i[0]
        if i[0] > max_x:
            max_x = i[0]
        if i[1] < min_y:
            min_y = i[1]
        if i[1] > max_y:
            max_y = i[1]
    face_height = max_y - min_y
    random_range = int(face_height * rand_radio)
    if need_random:
        min_x -= random.randint(0, random_range)
        min_y -= random.randint(0, random_range)
        max_x += random.randint(0, random_range)
        max_y += random.randint(0, random_range)
    if max_x > size[0]:
        max_x = size[0]
    if max_y > size[1]:
        max_y = size[1]
    return min_x, min_y, max_x, max_y


def read_all_annotation(path):
    files = os.listdir(path)
    annotation = {}
    for file in files:
        with open(path + "/" + file) as af:
            key = af.readline().strip()
            points = []
            annotation[key] = points
            for i in range(194):
                point = af.readline()
                sx, sy = point.split(',')
                x = float(sx)
                y = float(sy)
                points.append([x, y])
    return annotation

def bias_ann(ann, bias):
    for point in ann:
        point[0] = point[0] - bias[0]
        point[1] = point[1] - bias[1]

def crop_face_area(img, ann, need_random=True, rand_ratio=0.3):
    area = get_max_box(ann, img.size, need_random, rand_ratio)
    sub_img = img.crop(area)
    bias_ann(ann, (area[0], area[1]))
    return sub_img, ann


def pic_resize2square(img, des_size, ann, is_random=True):
    rows = img.height
    cols = img.width
    scale_rate = float(0)

    new_rows = des_size
    new_cols = des_size
    rand_x = 0
    rand_y = 0

    if rows > cols:
        scale_rate = des_size / rows
        new_cols = math.ceil(cols * scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        if is_random:
            rand_x = random.randint(0, math.floor(des_size - new_cols))
        else:
            rand_x = int(math.floor((des_size - new_cols) / 2))

    elif cols > rows:
        scale_rate = des_size / cols
        new_rows = math.ceil(rows * scale_rate)
        # print(rows, cols, new_rows, new_cols, scale_rate)
        if is_random:
            rand_y = random.randint(0, math.floor(des_size - new_rows))
        else:
            rand_y = int(math.floor((des_size - new_rows) / 2))
    else:
        scale_rate = des_size / cols
        new_rows = math.ceil(rows * scale_rate)

    new_img = img.resize((new_cols, new_rows))

    scaled_img = Image.new("RGB", (des_size, des_size))
    scaled_img.paste(new_img, box=(rand_x, rand_y))

    new_ann = []
    for point in ann:
        new_point = []
        for i in range(len(point)):
            new_point.append(point[i] * scale_rate)
        new_point[0] += rand_x
        new_point[1] += rand_y
        new_ann.append(new_point)

    return scaled_img, new_ann

def draw_ann(img, ann, font, font_size):
    draw = ImageDraw.Draw(img)
    new_ann = []
    for p in ann:
        new_ann.append(tuple(p))
    draw.point(tuple(new_ann),  fill=(255, 0, 0))
    for i in range(194):
        center_x = ann[i][0] - font_size/2
        center_y = ann[i][1] - font_size/2
        draw.text((center_x, center_y), str(i), fill=(0, 255, 0), font=font)


class HellenDataset(Dataset):
    def __init__(self, is_train, size):
        self.size = size
        self.is_train = is_train
        if is_train:
            self.file_path = cfg.path+cfg.train_txt
            self.pic_path = cfg.path+"train/"
        else:
            self.file_path = cfg.path + cfg.test_txt
            self.pic_path = cfg.path + "test/"
        self.annotations = read_all_annotation(cfg.path + cfg.annotation)
        self.files = []
        for line in open(self.file_path):
            self.files.append(line.replace('\n', ''))
        self.pic_strong = tfs.Compose([
            tfs.ColorJitter(0.5, 0.3, 0.3, 0.1),
            tfs.ToTensor()
        ])
        self.annotations = read_all_annotation(cfg.path + cfg.annotation)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file_name = self.files[item]
        img = Image.open(self.pic_path + file_name)
        ann = self.annotations[file_name.replace(".jpg", "")]
        sub_image, new_ann = crop_face_area(img, ann, rand_ratio=0.5)
        square_img, square_ann = pic_resize2square(sub_image, self.size, new_ann)
        return self.pic_strong(square_img), torch.from_numpy(np.array(square_ann)).float()


def test_dataset():
    font_size = 16
    font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)
    data_loader = DataLoader(dataset=HellenDataset(True, 600), batch_size=1, shuffle=True)
    transform = tfs.Compose([tfs.ToPILImage()])
    for i_batch, sample_batched in enumerate(data_loader):
        fig = plt.figure(num=1, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
        square_img = transform(sample_batched[0][0])
        draw_ann(square_img, sample_batched[1][0].numpy().tolist(), font1, font_size)
        plt.imshow(square_img)
        plt.show()
        plt.close()


# test_dataset()

def mean_face():
    font_size = 16
    font1 = ImageFont.truetype(r'./Ubuntu-B.ttf', font_size)
    helen = HellenDataset(True, 224)
    data_loader = DataLoader(dataset=helen, batch_size=1, shuffle=True)
    transform = tfs.Compose([tfs.ToPILImage()])
    mean_face = torch.zeros(3, 224, 224)
    mean_shape = torch.zeros(194, 2)
    for i_batch, sample_batched in enumerate(data_loader):
        mean_face += sample_batched[0][0]
        mean_shape += sample_batched[1][0]
    mean_face /= len(helen)
    mean_shape /= len(helen)
    fig = plt.figure(num=2, figsize=(15, 8), dpi=80)  # 开启一个窗口，同时设置大小，分辨率
    square_img = transform(mean_face)
    square_img.save("data/mean_face.jpg")
    draw_ann(square_img, mean_shape.numpy().tolist(), font1, font_size)
    square_img.save("data/mean_face_and_shape.jpg")
    np.savetxt("data/mean_shape", mean_shape.numpy())
    plt.imshow(square_img)
    plt.show()
    plt.close()


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    def __len__(self):
        return len(self.loader.dataset)

if __name__=='__main__':
    test_dataset()
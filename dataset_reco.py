import torchvision.transforms as transforms
import torch
import os
import cv2
import albumentations as A
import random
import glob2
import numpy as np
from PIL import Image, ImageDraw, ImageFont,ImageOps

from cv2_function import path2handimg, char2font

class Dataset(torch.utils.data.Dataset):

    def __init__(self, config):
        super().__init__()

        # ファイル全体をリストとして読み込み
        with open(config["use_char"]) as f:
            self.char_list = [s.strip() for s in f.readlines()]

        self.img_size = config["img_size"]

        self.transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])

        self.indices = []
        for char in self.char_list:
            path = "{}/{}.txt".format(config["use_root"], char)
            if os.path.exists(path):
                with open(path) as f:
                    char_paths = [s.strip() for s in f.readlines()]

                if config["save_num"] is not None:
                    char_paths = char_paths[:int(config["save_num"])]
                self.indices+=char_paths
        self.len = len(self.indices)

        # フォント
        with open("information/use_font.txt") as f:
            self.fontfile_paths = [s.strip() for s in f.readlines()]

    def path2info(self, path):
        char = path.split("/")[-3]
        label = self.char_list.index(char)
        label = torch.tensor(label)

        if int(random.uniform(0,1)+0.5):
            img = path2handimg(path, is_aug=True)
        else:
            count=0
            while True:
                char = path.split("/")[-3]
                fontfile = random.choice(self.fontfile_paths)
                img = char2font(char, fontfile, is_aug=True, is_pad=True)
                if img is not None:
                    break

                if count==20:
                    img = path2handimg(path, is_aug=True)
                    break
                count+=1
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = self.transforms(img)

        return img

    def __getitem__(self, index):
        path = self.indices[index]
        img = self.path2info(path)

        return img

    def __len__(self):
        return self.len
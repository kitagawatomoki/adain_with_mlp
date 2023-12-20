import torchvision.transforms as transforms
import torch
import os
import cv2
import albumentations as A
import random
import glob2
import numpy as np
from PIL import Image, ImageDraw, ImageFont,ImageOps

up_dot = ['"', "'", "^", "°","゜","「"]
under_dot = [',','，','.',',','。','、',"_", "」"]
center = ["・",'・',"一", "-", "ー", "―", "~","=", "丶"]
small = ["ぁ", "ぃ", "ぅ","ぇ","ぉ","っ","ゃ","ゅ","ょ","ゎ",
        "ァ", "ィ", "ゥ","ェ","ォ","ッ","ャ","ュ","ョ","ヮ","ㇲ","ㇻ"]

img_size = 256
rotate = A.Rotate(always_apply=False, 
                p=1, 
                limit=(-15, 15), 
                interpolation=0, 
                border_mode=0, 
                value=(0, 0, 0), 
                mask_value=None, 
                method='largest_box', 
                crop_border=False)

resize = A.Resize(height=img_size, width=img_size, p=1)
aug_list = [rotate, resize]
albumentations_aug = A.Compose(aug_list)

def char2font(char, fontfile, is_aug=False, is_pad=False):
    try:
        org_char = 224
        img = Image.new('RGB', ((org_char+30)*len(char), org_char+30), 'white')
        draw = ImageDraw.Draw(img)
        draw.font = ImageFont.truetype(fontfile, org_char)
        draw.text([0, -int(org_char*0.05)], char, (0, 0, 0))
        img = ImageOps.invert(img)

        img = img.crop(img.getbbox())
        img = ImageOps.invert(img)
        img = np.array(img)

        if np.sum(cv2.bitwise_not(img))==0 or np.sum(cv2.bitwise_not(img))>=255*img.shape[0]*img.shape[1]*3*0.95 and char!="一":
            return None
        else:
            if char not in up_dot + under_dot+ center+ small:

                # if int(random.uniform(0,1)+0.5):
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #     img = ocrodeg.random_blotches(img, 3e-4, 1e-4)
                #     img = (img*255).astype(np.uint8)
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # h, w , _ = img.shape
                # back_img = np.full((org_char, w, 3), 255, dtype=np.uint8)
                # if h < org_char:
                #     ymin = random.randint(0, org_char-h)
                # else:
                #     img = img[:org_char,:,:]
                #     ymin = 0

                # back_img[ymin:ymin+img.shape[0],:img.shape[1],:] = img
                # img = back_img

                img = adjust_square(img)
                if is_pad:
                    img = random_pad(img)

                # if int(random.uniform(0,1)+0.5):
                #     img = self.add_background_image(img)

                if is_aug:
                    img = cv2.bitwise_not(img)
                    transformed = albumentations_aug(image=img)
                    img = transformed["image"]
                    img = cv2.bitwise_not(img)

                return img
            else:
                if char in small:
                    img = adjust_square(img)
                    img = adjust_square(img, int((org_char - img.shape[1])/2))
                elif char in up_dot:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/2):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        back_img[:int(org_char/2),:int(org_char/2),:] = img
                        img = back_img
                    else:
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/2)-img.shape[0])
                        left_margin = random.randint(0, org_char-img.shape[1])

                        back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                        img = back_img

                    if char !="「":
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        back_img[:int(org_char/2),:,:] = img
                        img = back_img
                    else:
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/1.5)-img.shape[0])
                        back_img[up_margin:up_margin+img.shape[0],:,:] = img
                        img = back_img

                elif char in up_dot:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/2):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        back_img[:,int(org_char/2):,:] = img
                        img = back_img
                    else:
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/2)-img.shape[0])
                        left_margin = random.randint(0, org_char-img.shape[1])

                        back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                        img = back_img

                    if char !="」":
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        back_img[int(org_char/2):,:,:] = img
                        img = back_img
                    else:
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(int(org_char/1.5)-img.shape[0], int(org_char-img.shape[0]))
                        back_img[up_margin:up_margin+img.shape[0],:,:] = img
                        img = back_img

                elif char in center:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/1.5):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))

                    back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)

                    up_margin = random.randint(int(org_char/1.5)-img.shape[0], int(org_char)-img.shape[0])
                    up_margin = int(up_margin/2)
                    left_margin = random.randint(int(org_char/1.5)-img.shape[1], int(org_char)-img.shape[1])
                    left_margin = int(left_margin/2)

                    back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                    img = back_img

                return img
    except:
        return None

def path2handimg(path, is_aug = False):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # ETL9
    img = img[round(h*0.1):h - round(h*0.1), round(w*0.17):w - round(w*0.15)]
    # img = img[round(h*0.1):h - round(h*0.1), round(w*0.1):w - round(w*0.1)]
    img = cv2.resize(img,(img_size, img_size))
    ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img) # 白黒反転
    brank = 80
    img = cv2.copyMakeBorder(img, brank, brank, brank, brank, cv2.BORDER_CONSTANT)
    img = cv2.bitwise_not(img) # 白黒反転
    img = crop_img(img, brank)

    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if is_aug:
        img = cv2.bitwise_not(img)
        transformed = albumentations_aug(image=img)
        img = transformed["image"]
        img = cv2.bitwise_not(img)
    return img


def adjust_square(img, blank=0):
    h, w, _ = img.shape
    img = cv2.bitwise_not(img)
    if h>=w:
        img = cv2.copyMakeBorder(img, 0, 0, int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT, 0)
    else:
        img = cv2.copyMakeBorder(img, int((w-h)/2), int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
    # 正方形にする
    img = cv2.copyMakeBorder(img, blank, blank, blank, blank, cv2.BORDER_CONSTANT, 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (img.shape[0], img.shape[0]))

    return img

def random_pad(img, max_pad=40, pad_value=0):
    h, w, _ = img.shape
    img = cv2.bitwise_not(img)
    up_pad = random.randint(10, max_pad)
    down_pad = random.randint(10, max_pad)
    left_pad = random.randint(10, max_pad)
    right_pad = random.randint(10, max_pad)
    # 正方形にする
    img = cv2.copyMakeBorder(img, up_pad, down_pad, right_pad, left_pad, cv2.BORDER_CONSTANT, pad_value)
    img = cv2.bitwise_not(img)
    return img

def known_pad(img, pad=20, pad_value=0):
    h, w, _ = img.shape
    img = cv2.bitwise_not(img)
    up_pad, down_pad, left_pad, right_pad = pad, pad, pad, pad
    # 正方形にする
    img = cv2.copyMakeBorder(img, up_pad, down_pad, right_pad, left_pad, cv2.BORDER_CONSTANT, pad_value)
    img = cv2.bitwise_not(img)
    return img

def crop_img(img, brank):
    h_exit_pix = []
    w_exit_pix = []
    space = 10
    for i in range(img_size+brank*2):
        h_pix_sum = np.sum(img[i])
        w_pix_sum = np.sum(img[:, i])
        if h_pix_sum !=255*(img_size+brank*2):
            h_exit_pix.append(i)

        if w_pix_sum !=255*(img_size+brank*2):
            w_exit_pix.append(i)

    h_s = h_exit_pix[0]
    h_e = h_exit_pix[-1]
    w_s = w_exit_pix[0]
    w_e = w_exit_pix[-1]

    h = h_e - h_s
    w = w_e - w_s
    # print(h, w)

    if h > w:
        add_w = round((h - w)/2)
        w_s = w_s - add_w 
        w_e = w_e + add_w
    elif h < w:
        add_h = round((w - h)/2)
        h_s = h_s - add_h 
        h_e = h_e + add_h

    if h_s < space:
        h_s = space

    if h_e > img_size+brank*2 -space:
        h_e = img_size+brank*2 -space

    if w_s < space:
        w_s = space

    if w_e > img_size+brank*2 -space:
        w_e = img_size+brank*2 -space

    img = img[h_s-space: h_e+space, w_s-space: w_e+space]
    img = cv2.resize(img,(img_size, img_size))

    return img



def rote_img(img, angle=45):
    h, w, _ = img.shape
    center = (int(w/2), int(h/2))
    trans = cv2.getRotationMatrix2D(center, angle , 1)
    img = cv2.bitwise_not(img)
    img = cv2.warpAffine(img, trans, (w,h))
    img = cv2.bitwise_not(img)
    return img


import json

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

kanji2element = load_json("../../diffusion_models2/CLIP/output_jp_new_all.json")
with open("../../diffusion_models2/CLIP/vocabulary_cjkvi_new_all.txt") as f:
    vocabulary_list = [s.strip() for s in f.readlines()]
vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

def get_tokens(char):
    tokens = []
    tokens_tensor = []

    elements = kanji2element[char]
    element = random.choice(elements)
    tokens = element.split(" ")
    # print(char, tokens)

    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    token_type_ids = [0]*len(tokens)
    attention_mask = [0 if t == "[PAD]" else 1 for t in tokens]
    tokens_tensor = [vocabulary_list.index(t) if t in vocabulary_list else 1 for t in tokens]

    token_type_ids_tensor = torch.tensor(token_type_ids)
    attention_mask_tensor = torch.tensor(attention_mask)
    tokens_tensor = torch.tensor(tokens_tensor)

    # inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

    return tokens, tokens_tensor,token_type_ids_tensor,attention_mask_tensor
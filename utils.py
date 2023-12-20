import json

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

import numpy as np

def make_grid(imgs, nrow, padding=0):
    """Numpy配列の複数枚の画像を、1枚の画像にタイルします

    Arguments:
        imgs {np.ndarray} -- 複数枚の画像からなるテンソル
        nrow {int} -- 1行あたりにタイルする枚数

    Keyword Arguments:
        padding {int} -- グリッドの間隔 (default: {0})

    Returns:
        [np.ndarray] -- 3階テンソル。1枚の画像
    """
    assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)
    # border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0)) # 下と右だけにpaddingを入れる
        height += padding
        width += padding
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:
        x = x[:(height * ncol - padding),:(width * nrow - padding),:] # 右端と下端のpaddingを削除
    return x

import torch

def return_img(image):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    image = image.to("cpu")

    x = image.mul(torch.FloatTensor(IMAGENET_STD).view(3, 1, 1))
    x = x.add(torch.FloatTensor(IMAGENET_MEAN).view(3, 1, 1)).detach().numpy()
    x = np.transpose(x, (0, 2, 3, 1))

    image = np.clip(x, 0, 1)
    image = (image*255).astype(np.uint8)

    return image

from torch.nn.utils.rnn import pad_sequence

#ミニバッチを取り出して長さを揃える関数
def img_collate_func(batch):
    imgs1,imgs2, labels, xs, ys, zs = [], [], [], [], [],[]
    for img1, img2, label, x,y,z in batch:
        imgs1.append(img1)
        imgs2.append(img2)
        labels.append(label)
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        zs.append(torch.LongTensor(z))

    #データ長を揃える処理
    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    zs = pad_sequence(zs, batch_first=True, padding_value=0.0)
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    labels = torch.stack(labels)
    return imgs1, imgs2, labels, xs, ys, zs

import torch
import torch.nn.functional as F
from torch import nn
from model_clip import CLIPDualEncoderModel

class CLIP_fn(nn.Module):
    def __init__(self, trainable: bool = False) -> None:
        super().__init__()

        # ファイル全体をリストとして読み込み
        with open("information/vocabulary_cjkvi_fixed.txt") as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        image_encoder_alias = "resnet50"
        self.model = CLIPDualEncoderModel(image_encoder_alias, len(vocabulary_list))
        model_path = "pretraind_weight/CLIP.pt"
        weight = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(weight)
        for param in self.model.parameters():
            param.requires_grad = trainable
        self.model.to("cpu")
        self.model.eval()

        self.temperature = 1.0
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def compute_losses(self, image, inputs):
        image_embeddings, text_embeddings = self.model(image,inputs)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0


def calculate_norm(r_style, l_style):
    # MSE = nn.MSELoss()
    return torch.mean(torch.cdist(r_style, l_style, p=2))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple
import os
import hashlib
import requests
import numpy as np
from tqdm import tqdm

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None, discriminator_weight=1.0):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, device, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer(device)
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False).to(device)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout).to(device)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout).to(device)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout).to(device)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout).to(device)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout).to(device)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class ScalingLayer(nn.Module):
    def __init__(self, device):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None].to(device))
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None].to(device))

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
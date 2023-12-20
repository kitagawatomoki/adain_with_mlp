import torch
from torch import nn

from functools import partial
from einops import rearrange, reduce, repeat

from model_utils import (ResnetBlock, Residual, PreNorm,
                        LinearAttention, Attention,
                        Downsample, Upsample,
                        prob_mask_like)
from model_clip import Trained_CLIP_TextEncoder
from model import AutoEncoder

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class AdaIN_with_CLIP(nn.Module):
    def __init__(
        self,
        dim,
        cond_drop_prob = 0.5,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        style_encoder_weight_path=None
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4
        self.cond_drop_prob = cond_drop_prob

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # AdaIN layers
        self.adain = adaptive_instance_normalization

        # class embeddings
        self.textencoder = Trained_CLIP_TextEncoder()
        self.null_classes_emb = nn.Parameter(torch.randn(256))
        self.post_textencoder = nn.Sequential(
            nn.Linear(256, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine content encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)

        # determine style encoder
        self.style_encoder = AutoEncoder(
            dim = dim,
            dim_mults = dim_mults
        )
        if style_encoder_weight_path is not None:
            weight = torch.load(style_encoder_weight_path)
            self.style_ae.load_state_dict(weight)

        # determine decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        self.out_dim = channels
        self.final_res_block = block_klass(dim, dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def content_encoder(self, x, c):
        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, c)

        return x

    def decoder(self, x, c):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, c)
        return self.final_conv(x)

    def post_classes_mlp(self, x, classes):
        batch, device = x.shape[0], x.device
        classes_emb = self.textencoder(classes)

        if self.cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - self.cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.post_textencoder(classes_emb)

        return c

    def encode(self, x, style_image, classes):
        c = self.post_classes_mlp(x, classes)

        x  = self.content_encoder(x, c)
        s_x = self.style_encoder(style_image)

        x = self.adain(x, s_x)

        return x, s_x, c

    def decode(self, x, c):
        x = self.u_net_decoder(x, c)

        return x

    def forward(self,x,style_image,classes):
        x, _, c = self.encode(x, style_image, classes)
        x = self.decode(x, c)

        return x
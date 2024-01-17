import torch
from torch import nn

from functools import partial

from model_utils import (ResnetBlock, Residual, PreNorm,
                        LinearAttention, Attention,
                        Downsample, Upsample,
                        ActNorm)

######################
#### AutoEncoder #####
######################
class AutoEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine encoder
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

    def encoder(self, x):
        x = self.init_conv(x)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, None)
            x = block2(x, None, None)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, None)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, None)
        return x

    def decoder(self, x):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, None)
            x = block2(x, None, None)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, None)
        return self.final_conv(x)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

########################
#### Discriminator #####
########################
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

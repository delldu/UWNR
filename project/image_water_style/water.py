"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class downsample(nn.Module):
    def __init__(self, in_ch, out_ch, conv=nn.Conv2d, act=nn.ELU):
        super(downsample, self).__init__()
        self.mpconv = nn.Sequential(
            conv(in_ch, out_ch, kernel_size=3, stride=2, padding=3 // 2), act(inplace=True), BasicBlock(out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class upsample(nn.Module):
    def __init__(self, in_ch, conv=nn.Conv2d, act=nn.ELU):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            conv(in_ch, 2 * in_ch, kernel_size=3, stride=1, padding=3 // 2),
            nn.ELU(),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        y = self.up(x)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, chns, factor):
        super(SpatialAttention, self).__init__()
        self.spatial_pool = nn.Sequential(
            nn.Conv2d(chns, chns // factor, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(chns // factor, 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        spatial_map = self.spatial_pool(x)
        return x * spatial_map


class ChannelAttention(nn.Module):
    def __init__(self, chns, factor):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_map = nn.Sequential(
            nn.Conv2d(chns, chns // factor, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(chns // factor, chns, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        map = self.channel_map(avg_pool)
        return x * map


class BasicBlock(nn.Module):
    def __init__(self, chns):
        super(BasicBlock, self).__init__()
        self.conk3 = nn.Conv2d(chns, chns, 3, 1, 3 // 2)
        self.conk1 = nn.Conv2d(chns, chns, 1, 1, 1 // 2)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.SA = SpatialAttention(chns, 4)
        self.CA = ChannelAttention(chns, 4)
        self.norm = nn.InstanceNorm2d(chns // 2, affine=True)

    def forward(self, x):
        residual = x
        y = self.conk1(x) + self.conk3(x) + residual

        output = self.leakyrelu(y)
        output = self.CA(self.SA(output)) + residual
        return output


class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        # Define max GPU/CPU memory -- GPU 3G, 64ms
        self.MAX_H = 1024
        self.MAX_W = 2048
        self.MAX_TIMES = 16

        self.in_conv_down1 = downsample(7, 64)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        self.down4 = downsample(256, 512)

        self.up1 = upsample(512)
        self.up2 = upsample(256)
        self.up3 = upsample(128)
        self.up4 = upsample(64)

        self.out = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self.load_weights()


    def load_weights(self, model_path="models/image_water.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        if os.path.exists(checkpoint):
            self.load_state_dict(torch.load(checkpoint))

    def forward(self, x):
        # # input = content_tensor + style_tensor
        # content_tensor = input[:, 0:3, :, :]
        # style_tensor = input[:, 3:6, :, :]

        # # x = content_tensor + depth_tensor + style_tensor_map
        # depth_tensor = content_tensor[:, 0:1, :, :]
        # style_tensor_map = style_tensor        
        # x = torch.cat((content_tensor, depth_tensor, style_tensor_map), dim=1)

        # Orignal inference
        residual = x[:, 0:3, :, :]
        x2 = self.in_conv_down1(x)
        x4 = self.down2(x2)
        x8 = self.down3(x4)
        x16 = self.down4(x8)

        y = x16
        y8 = self.up1(y) + x8
        y4 = self.up2(y8) + x4
        y2 = self.up3(y4) + x2
        y = self.up4(y2)

        out = self.out(y)
        return out

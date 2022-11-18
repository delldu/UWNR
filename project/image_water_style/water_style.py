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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from . import depth
from . import water

import pdb

class GaussFilter(nn.Module):
    """
    3x3 Guassian filter
    """
    def __init__(self):
        super(GaussFilter, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)

        # self.conv.bias.data.fill_(0.0)
        self.conv.weight.data.fill_(0.0625)
        self.conv.weight.data[:, :, 0, 1] = 0.125
        self.conv.weight.data[:, :, 1, 0] = 0.125
        self.conv.weight.data[:, :, 1, 2] = 0.125
        self.conv.weight.data[:, :, 2, 1] = 0.125
        self.conv.weight.data[:, :, 1, 1] = 0.25

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.interpolate(x, (H//32, W//32), mode="bilinear", align_corners=True)
        for i in range(3):
            x = self.conv(x)
        x = (x - x.min())/(x.max() - x.min() + 1e-8)
        x = F.interpolate(x, (H, W), mode="bilinear", align_corners=True)
        return x


class ImageWaterStyleModel(nn.Module):
    def __init__(self):
        super(ImageWaterStyleModel, self).__init__()
        # Define max GPU/CPU memory -- GPU 3G, 64ms
        self.MAX_H = 1024
        self.MAX_W = 2048
        self.MAX_TIMES = 16

        self.depth_model = depth.MegaDepthModel().eval()
        self.water_model = water.GeneratorModel().eval()
        self.gauss_filter = GaussFilter()

        torch.save(self.state_dict(), "/tmp/image_water_style.pth")
        # self.load_weights()

    def load_weights(self, model_path="models/image_water_style.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))

    def forward(self, input):
        # input = content_tensor + style_tensor
        content_tensor = input[:, 0:3, :, :]
        style_tensor = input[:, 3:6, :, :]

        # x = content_tensor + depth_tensor + style_tensor_map
        depth_tensor = self.depth_model(content_tensor)
        style_tensor_map = self.gauss_filter(style_tensor)

        x = torch.cat((content_tensor, depth_tensor, style_tensor_map), dim=1)

        return self.water_model(x)

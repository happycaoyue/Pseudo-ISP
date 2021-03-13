# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
## The reproduction of PyTorch version of CBDNet is done by me, which is same as number of parameters and network structure of author official MatConvNet version
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, channels = 3, filters_num = 32, filters_level1 = 64, filters_level2 = 128, filters_level3 = 256):
        super(Net, self).__init__()

        self.est = nn.Sequential(
            nn.Conv2d(channels, filters_num, 3, 1, 1), # E_x01
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1), # E_x02
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1), # E_x03
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1), # E_x04
            nn.ReLU(True),
            nn.Conv2d(filters_num, channels, 3, 1, 1), # E_x05
            nn.ReLU(True))

        self.down1 = nn.Sequential(
            nn.Conv2d(channels * 2, filters_level1, 3, 1, 1),   # DS01_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level1, filters_level1, 3, 1, 1), # DS01_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level1, filters_level1, 3, 1, 1), # DS01_x02
            nn.ReLU(True),
            nn.Conv2d(filters_level1, filters_level1, 3, 1, 1), # DS01_x03
            nn.ReLU(True))

        # sum 2
        self.down2 = nn.Sequential(
            nn.Conv2d(in_channels=filters_level1, out_channels=filters_level1 * 4, kernel_size=2, stride=2, padding=0), # DS02_sx
            nn.Conv2d(in_channels=filters_level1 * 4, out_channels=filters_level2, kernel_size=1, stride=1, padding=0), # DS02_x00_cf
            nn.Conv2d(filters_level2, filters_level2, 3, 1, 1),  # DS02_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level2, filters_level2, 3, 1, 1),  # DS02_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level2, filters_level2, 3, 1, 1),  # DS02_x02
            nn.ReLU(True))

        # sum 3
        self.down3 = nn.Sequential(
            nn.Conv2d(in_channels=filters_level2, out_channels=filters_level2 * 4, kernel_size=2, stride=2, padding=0), # DS03_sx 128 - 512
            nn.Conv2d(in_channels=filters_level2 * 4, out_channels=filters_level3, kernel_size=1, stride=1, padding=0), # DS03_x00_cf 512 - 256
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # DS03_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # DS03_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # DS03_x02
            nn.ReLU(True))

        self.up3 = nn.Sequential(
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # UPS03_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # UPS03_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level3, filters_level3, 3, 1, 1),  # UPS03_x02
            nn.ReLU(True),
            nn.Conv2d(filters_level3, filters_level3 * 2, 3, 1, 1),  # UPS03_x03
            nn.ReLU(True),
            nn.ConvTranspose2d(filters_level3 * 2, filters_level2, 2, stride=2) # USP02_x00
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(filters_level2, filters_level2, 3, 1, 1),  # US02_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level2, filters_level2, 3, 1, 1),  # US02_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level2, filters_level2 * 2, 3, 1, 1),  # US02_x02
            nn.ReLU(True),
            nn.ConvTranspose2d(filters_level2 * 2, filters_level1, 2, stride=2) # USP01_x00
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(filters_level1, filters_level1, 3, 1, 1),  # US01_x00
            nn.ReLU(True),
            nn.Conv2d(filters_level1, filters_level1, 3, 1, 1),  # US01_x01
            nn.ReLU(True),
            nn.Conv2d(filters_level1, channels, 3, 1, 1)  # US01_x02
      )




    def forward(self, inputs):

        c0 = inputs
        est = self.est(inputs)

        c1 = self.down1(torch.cat([inputs, est], dim=1))
        c2 = self.down2(c1)
        c3 = self.down3(c2)
        c4 = self.up3(c3)
        c5 = self.up2(c4 + c2)
        c6 = self.up1(c5 + c1)


        return (c6 + c0)


# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
## The reproduction of PyTorch version of MWCNN
## The DWT and IDWT operation implementation are done by me, which is slightly different from the author's official version.
"""
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as init
import torch.nn.functional as F

class DWTForward(nn.Module):

    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y




class prev_block(nn.Module):
    '''conv => BN => ReLU+(conv => BN => ReLU) * 3'''
    def __init__(self, in_ch, out_ch):
        super(prev_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class next_block(nn.Module):
    '''(conv => BN => ReLU) * 3 + conv => BN => ReLU'''

    def __init__(self, in_ch, out_ch):
        super(next_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class level1_prev_block(nn.Module):
    '''conv => ReLU+(conv => BN => ReLU) * 3'''
    def __init__(self, in_ch, out_ch):
        super(level1_prev_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class level1_next_block(nn.Module):
    '''(conv => BN => ReLU) * 3 + conv '''

    def __init__(self, in_ch, out_ch):
        super(level1_next_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        in_channels = 3
        dwt1_num = 12

        level1_num = 160
        level2_num = 256
        level3_num = 256

        self.level1_prev = level1_prev_block(dwt1_num, level1_num)
        self.level1_next = level1_next_block(level1_num, dwt1_num)

        self.level2_prev = prev_block(level1_num * 4, level2_num)
        self.level2_next = next_block(level2_num, level1_num * 4)

        self.level3_prev = prev_block(level2_num * 4, level3_num)
        self.level3_next = next_block(level3_num, level2_num * 4)

        self._initialize_weights()
        self.DWT = DWTForward()
        self.IWT = DWTInverse()

    def forward(self, x):
        # 1*4
        dwt1 = self.DWT(x)
        # 4*160
        level1_prev_block = self.level1_prev(dwt1)

        dwt2 = self.DWT(level1_prev_block)
        level2_prev_block = self.level2_prev(dwt2)

        dwt3 = self.DWT(level2_prev_block)
        level3_prev_block = self.level3_prev(dwt3)

        level3_next_block = self.level3_next(level3_prev_block)
        sum3 = torch.add(dwt3, level3_next_block)
        iwt3 = self.IWT(sum3)

        level2_next_block = self.level2_next(iwt3)
        sum2 = torch.add(dwt2, level2_next_block)
        iwt2 = self.IWT(sum2)

        level1_next_block = self.level1_next(iwt2)
        sum1 = torch.add(dwt1, level1_next_block)
        out = self.IWT(sum1)


        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
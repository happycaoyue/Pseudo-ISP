# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class HITVPCTeam:
    # conv - relu - conv - sum
    class RB(nn.Module):
        def __init__(self, filters):
            super(HITVPCTeam.RB, self).__init__()
            self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.act = nn.ReLU(True)
            self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)

        def forward(self, x):
            c0 = x
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            return x + c0

    class NRB(nn.Module):
        def __init__(self, n, f):
            super(HITVPCTeam.NRB, self).__init__()
            nets = []
            for i in range(n):
                nets.append(HITVPCTeam.RB(f))
            self.body = nn.Sequential(*nets)

        def forward(self, x):
            return self.body(x)

    class DWTForward(nn.Module):

        def __init__(self):
            super(HITVPCTeam.DWTForward, self).__init__()
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
            super(HITVPCTeam.DWTInverse, self).__init__()
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


class Net(nn.Module):
    def __init__(self, channels = 3, filters_level1 = 160, filters_level2 = 256, filters_level3 = 512, n_rb=4):
        super(Net, self).__init__()


        self.head = HITVPCTeam.DWTForward()

        self.down1 = nn.Sequential(
            nn.Conv2d(channels * 4, filters_level1, 3, 1, 1),
            nn.ReLU(True),
            HITVPCTeam.NRB(n_rb, filters_level1))

        self.level2_head = nn.Sequential(
            nn.Conv2d(channels * 16, filters_level2, 3, 1, 1),
            nn.ReLU(True))

        self.level1_down = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level1 * 4, filters_level2, 3, 1, 1),
            nn.ReLU(True))

        # sum 2
        self.down2 = nn.Sequential(
            nn.Conv2d(filters_level2 * 2, filters_level2, 3, 1, 1),
            nn.ReLU(True),
            HITVPCTeam.NRB(n_rb, filters_level2))

        self.level3_head = nn.Sequential(
            nn.Conv2d(channels * 64, filters_level3, 3, 1, 1),
            nn.ReLU(True))

        self.level2_down = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level2 * 4, filters_level3, 3, 1, 1),
            nn.ReLU(True))


        self.down3 = nn.Sequential(
            nn.Conv2d(filters_level3 * 2, filters_level3, 3, 1, 1),
            nn.ReLU(True))

        self.middle = HITVPCTeam.NRB(n_rb * 2, filters_level3)



        self.up1 = nn.Sequential(
            nn.Conv2d(filters_level3, filters_level2 * 4, 3, 1, 1),
            nn.ReLU(True),
            HITVPCTeam.DWTInverse())


        self.level2_middle = HITVPCTeam.NRB(n_rb, filters_level2)

        self.up2 = nn.Sequential(
            nn.Conv2d(filters_level2, filters_level1 * 4, 3, 1, 1),
            nn.ReLU(True),
            HITVPCTeam.DWTInverse())


        self.up3 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level1),
            nn.Conv2d(filters_level1, channels * 4, 3, 1, 1))

        self.tail = HITVPCTeam.DWTInverse()

    def forward(self, inputs):
        c0 = inputs

        dwt1 = self.head(c0)
        dwt2 = self.head(dwt1)
        dwt3 = self.head(dwt2)

        c2 = self.down1(dwt1)

        c3 = self.down2(torch.cat([self.level1_down(c2), self.level2_head(dwt2)], dim=1))
        c4 = self.down3(torch.cat([self.level2_down(c3), self.level3_head(dwt3)], dim=1))
        m3 = self.middle(c4)
        c5 = self.up1(m3) + c3
        m2 = self.level2_middle(c5)
        c6 = self.up2(m2) + c2
        c7 = self.up3(c6) + dwt1

        return self.tail(c7)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


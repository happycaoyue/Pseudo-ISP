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
import torch.nn.init as init

def pixel_unshuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, out_height, upscale_factor,
        out_width, upscale_factor)

    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)

class PixelUnShuffle(nn.Module):

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
# batchsize * C * patchsize * patchsize  16 * 3 * 80 * 80
# 16 * 3 * 80 * 80
# red 16 * 1 * 40 * 40
# green_red 16 * 1 * 40 * 40
# green_blue 16 * 1 * 40 * 40
# blue 16 * 1 * 40 * 40
# output 16 * 4 * 40 * 40
def mosaic(images):
  """Extracts RGGB Bayer planes from RGB image."""
  red = images[:, 0, 0::2, 0::2]
  green_red = images[:, 1, 0::2, 1::2]
  green_blue = images[:, 1, 1::2, 0::2]
  blue = images[:, 2, 1::2, 1::2]
  output = torch.stack((red, green_red, green_blue, blue), dim=1)
  return output

class Mosaic_Operation(nn.Module):

    def __init__(self):
        super(Mosaic_Operation, self).__init__()


    def forward(self, input):
        return mosaic(input)


class RGB2PACK(nn.Module):
    def __init__(self, channels=3, filters_num = 128, filters_pack = 4):
        super(RGB2PACK, self).__init__()

        # RGB2RAW Network
        self.RGB2RAW = nn.Sequential(
            nn.Conv2d(channels, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, channels, 3, 1, 1),
            nn.ReLU(True))

        # Mosaic
        self.mosaic = Mosaic_Operation()
        self._initialize_weights()
        # # Pack Raw
        # self.Pack_Raw = PixelUnShuffle(upscale_factor=2)

    def forward(self, X_rgb, Y_rgb):

        # RGB2RAW
        X_dem = self.RGB2RAW(X_rgb)
        Y_dem = self.RGB2RAW(Y_rgb)

        # mosaic
        X_pack = self.mosaic(X_dem)
        Y_pack = self.mosaic(Y_dem)


        # Pack Raw
        # X_pack = self.Pack_Raw(X_raw)
        # Y_pack = self.Pack_Raw(Y_raw)

        return X_pack, Y_pack
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

class PACK2RGB(nn.Module):
    def __init__(self, channels=3, filters_num = 128, filters_pack = 4):
        super(PACK2RGB, self).__init__()

        # RAW2RGB Network
        self.RAW2RGB = nn.Sequential(
            nn.Conv2d(filters_pack, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, channels * 4, 3, 1, 1),
            nn.ReLU(True))

        # Pixel Shuffle Upsampling
        self.PS_U = nn.PixelShuffle(2)
        self._initialize_weights()



    def forward(self, X_pack, Y_pack):

        # RAW2RGB
        X = self.RAW2RGB(X_pack)
        Y = self.RAW2RGB(Y_pack)

        # Pixel Shuffle Upsampling
        X_output = self.PS_U(X)
        Y_output = self.PS_U(Y)

        return X_output, Y_output
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
# nn.Conv2d(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True))
class Noise_Model_Network(nn.Module):
    def __init__(self, channels=3, filters_num = 128, filters_pack = 4):
        super(Noise_Model_Network, self).__init__()

        # Noise Model Network
        self.network = nn.Sequential(
            nn.Conv2d(filters_pack, filters_num, 1, 1, 0, groups=1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_num, 1, 1, 0, groups=1),
            nn.ReLU(True),
            nn.Conv2d(filters_num, filters_pack, 1, 1, 0, groups=1),
            nn.ReLU(True))
        self._initialize_weights()

    def forward(self, X_pack):
        Output = self.network(X_pack)
        return Output
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


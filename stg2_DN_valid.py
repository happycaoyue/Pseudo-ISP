# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
"""
import os
import random
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import cv2
from net.CBDNet import Net
import math
random.seed()
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def findLastCheckpoint(save_dir, save_pre):
    file_list = glob.glob(os.path.join(save_dir, save_pre + '*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*" + save_pre +"(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def DN_valid(args, noisy_valid, denoised_valid):

    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre = args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'


    # net architecture
    dn_net = Net()

    # Move to GPU
    dn_model = nn.DataParallel(dn_net).cuda()

    # load old params, optimizer, state
    if args.resume == "continue":
        tmp_ckpt = torch.load(args.last_ckpt)
        training_params = tmp_ckpt['training_params']
        start_epoch = initial_epoch + 1

        # Initialize dn_model
        pretrained_dict = tmp_ckpt['state_dict']
        model_dict=dn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dn_model.load_state_dict(model_dict)
    if args.resume=="continue":

        print("Evaluating last epoch FIRST!")
        print("Initial model val psnr: \n")

        dn_model.eval()
        s = sio.loadmat('test_epoch_psnr.mat')
        psnr_data = s["tep"]
        psnr_all = np.array([0.0, 0.0, 0.0], np.float32)
        ssim_all = np.array([0.0, 0.0, 0.0], np.float32)
        ssim_x_ = 0
        psnr = 0
        psnr_val = 0
        ssim_val = 0
        valid_num = noisy_valid.shape[0]

        with torch.no_grad():
            psnr_val = 0
            ssim_val = 0

            for ii in range(valid_num):
                print(ii)

                Inoisy_crop = np.transpose(noisy_valid[ii], (1, 2, 0))
                Iclean_crop = np.transpose(denoised_valid[ii], (1, 2, 0))


                origin = Inoisy_crop.copy()
                noisy = Inoisy_crop.copy()
                noisy_data = []
                out_data = []
                out_data_real = []

                out_ = np.zeros(origin.shape)
                output = np.zeros(origin.shape)
                noisy_ = np.zeros((origin.shape[2], origin.shape[0], origin.shape[1]))
                temp1 = np.zeros((origin.shape[2], origin.shape[1], origin.shape[0]))
                temp2 = np.zeros((origin.shape[2], origin.shape[0], origin.shape[1]))
                temp3 = np.zeros((origin.shape[2], origin.shape[1], origin.shape[0]))

                # (3, 256, 256)
                for a in range(3):
                    noisy_[a, :, :] = noisy[:, :, a]

                # rotate / flip
                noisy_data.append(noisy_)
                for a in range(3):
                    temp1[a, :, :] = np.rot90(noisy_[a, :, :], 1)
                    temp2[a, :, :] = np.rot90(noisy_[a, :, :], 2)
                    temp3[a, :, :] = np.rot90(noisy_[a, :, :], 3)
                noisy_data.append(temp1)
                noisy_data.append(temp2)
                noisy_data.append(temp3)

                noisy_data.append(np.fliplr(noisy_data[0]).copy())
                noisy_data.append(np.fliplr(noisy_data[1]).copy())
                noisy_data.append(np.fliplr(noisy_data[2]).copy())
                noisy_data.append(np.fliplr(noisy_data[3]).copy())

                for x in range(8):
                    img = np.expand_dims(noisy_data[x], 0)
                    input = torch.tensor(img).cuda().float()
                    with torch.no_grad():
                        out = dn_model(input)
                    out_data.append(out.cpu().data[0].numpy().astype(np.float32))

                for a in range(8):
                    out_data_real.append(np.zeros((origin.shape[2], origin.shape[0], origin.shape[1])))

                out_data[4] = np.fliplr(out_data[4])
                out_data[5] = np.fliplr(out_data[5])
                out_data[6] = np.fliplr(out_data[6])
                out_data[7] = np.fliplr(out_data[7])

                for a in range(3):
                    out_data_real[1][a, :, :] = np.rot90(out_data[1][a, :, :], -1)
                    out_data_real[2][a, :, :] = np.rot90(out_data[2][a, :, :], -2)
                    out_data_real[3][a, :, :] = np.rot90(out_data[3][a, :, :], -3)

                    out_data_real[5][a, :, :] = np.rot90(out_data[5][a, :, :], -1)
                    out_data_real[6][a, :, :] = np.rot90(out_data[6][a, :, :], -2)
                    out_data_real[7][a, :, :] = np.rot90(out_data[7][a, :, :], -3)

                out_data_real[0] = out_data[0]
                out_data_real[4] = out_data[4]

                for x in range(8):
                    for a in range(3):
                        out_[:, :, a] = out_data_real[x][a, :, :]
                    output += out_
                output /= 8.0
                output[output < 0] = 0
                output[output > 1] = 1.0

                psnr = calculate_psnr(output * 255.0, Iclean_crop * 255.0)
                ssim_x_ = calculate_ssim(output * 255.0, Iclean_crop * 255.0)
                psnr_val += psnr
                ssim_val += ssim_x_

            psnr_val /= valid_num
            ssim_val /= valid_num
            psnr_all[0] = psnr_val
            ssim_all[0] = ssim_val

        psnr_data = np.column_stack((psnr_data, [start_epoch-1, psnr_all[0], ssim_all[0], psnr_all[1], ssim_all[1], psnr_all[2], ssim_all[2]]))
        s["tep"] = psnr_data
        sio.savemat('test_epoch_psnr.mat', s)





# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
"""
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import scipy.io
from stg2_DN_options import opt
from net.CBDNet import Net

def main(args):

    initial_epoch = 51
    save_dir = './submit_DND_CBDNet_e' + str(initial_epoch) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = './net_last_ckpt/' + args.save_prefix + str(initial_epoch) + '.pth'
    # net architecture
    dn_net = Net()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        dn_model = nn.DataParallel(dn_net).cuda()
    else:
        dn_model = nn.DataParallel(dn_net).cuda()
    # Optimizer

    # load old params, state
    if args.resume == "continue":
        tmp_ckpt=torch.load(args.last_ckpt)

        # Initialize dn_model
        pretrained_dict = tmp_ckpt['state_dict']
        model_dict=dn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dn_model.load_state_dict(model_dict)

    if args.resume=="continue":

        dn_model.eval()

        for n in range(50):
            for k in range(20):
                print("Cell No. : %d\n" % (k + 1))
                # 512 * 512 * 3
                noisy_valid_dir = '../dataset/DND_mat_image/' + 'noisy_mat/%04d_%02d.mat' % (n + 1, k + 1)
                print(noisy_valid_dir)
                mat_file = sio.loadmat(noisy_valid_dir)
                # get input numpy
                Inoisy_crop = np.float32(np.array(mat_file['Inoisy_crop']))

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

                output = np.float32(output)

                save_file = save_dir + '%04d_%02d.mat' % (n + 1, k + 1)
                scipy.io.savemat(save_file, {'Idenoised_crop': output})


if __name__ == "__main__":

    main(opt)

    exit(0)




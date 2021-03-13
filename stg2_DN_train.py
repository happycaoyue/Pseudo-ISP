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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import scipy.io as sio
from net.pseudoisp import RGB2PACK, PACK2RGB, Noise_Model_Network
from skimage.io import imread
from stg2_DN_options import opt
from h5_files.h5_dataset import Dataset # for loading .h5 data
import math
import h5py
import stg2_DN_valid
# You can also choose other networks MWCNN RIDNet PT-MWRN and so on
from net.CBDNet import Net
random.seed()
def get_gaussian_kernel(kernel_size=21, sigma=5, channels=3):
    #if not kernel_size: kernel_size = int(2*np.ceil(2*sigma)+1)
    #print("Kernel is: ",kernel_size)
    #print("Sigma is: ",sigma)
    padding = kernel_size//2
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter, padding


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

def data_augmentation(image, mode):
    r"""Performs dat augmentation of the input image

    Args:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
            0 - no transformation
            1 - flip up and down
            2 - rotate counterwise 90 degree
            3 - rotate 90 degree and flip up and down
            4 - rotate 180 degree
            5 - rotate 180 degree and flip
            6 - rotate 270 degree
            7 - rotate 270 degree and flip
    """
    out = image.copy()
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out
# Synthesis Dataset Generation
def Generate_Synthesis_Dataset(args):
    # Synthesis
    # args.traindbf = './dataset_h5py/Synthesis_Div_p160_s60_srgb_uint8.h5'
    train_num = 0
    patch_size = 512
    patchsize = 160
    if os.path.exists(args.traindbf):
        os.remove(args.traindbf)
    else:
        print('no such file:%s' % args.traindbf)
    with h5py.File(args.traindbf, 'w') as h5f:
        # 100 no same random number
        resultList = random.sample(range(1, 800), 200)
        for tt in range(200):
            image_num = resultList[tt]
            image_name = '%04d.png' % (image_num)
            gt = np.array(imread(args.Div_path + image_name), dtype=np.uint8)
            if not os.path.exists(args.Div_path + image_name):
                print("Please download DIV2K datase")
                assert (os.path.exists(args.Div_path + image_name))
            if gt.shape[2] != 3:
                image_num = random.randint(1, 800)
                image_name = '%04d.png' % (image_num)
                gt = np.array(imread(args.Div_path + image_name), dtype=np.uint8)
            if gt.shape[2] != 3:
                image_num = random.randint(1, 800)
                image_name = '%04d.png' % (image_num)
                gt = np.array(imread(args.Div_path + image_name), dtype=np.uint8)
            if gt.shape[2] != 3:
                image_num = random.randint(1, 800)
                image_name = '%04d.png' % (image_num)
                gt = np.array(imread(args.Div_path + image_name), dtype=np.uint8)
            if gt.shape[2] != 3:
                image_num = random.randint(1, 800)
                image_name = '%04d.png' % (image_num)
                gt = np.array(imread(args.Div_path + image_name), dtype=np.uint8)

            img_w = gt.shape[0]
            img_h = gt.shape[1]
            if img_h > 511 and img_w > 511 and gt.shape[2] == 3:
                w_num = int(np.ceil(img_w / patch_size))
                h_num = int(np.ceil(img_h / patch_size))
                for w_index in range(w_num):
                    for h_index in range(h_num):
                        start_x = w_index * patch_size
                        end_x = start_x + patch_size - 1
                        if end_x > img_w - 1:
                            end_x = img_w - 1
                            start_x = end_x - patch_size + 1
                        start_y = h_index * patch_size
                        end_y = start_y + patch_size - 1
                        if end_y > img_h - 1:
                            end_y = img_h - 1
                            start_y = end_y - patch_size + 1
                        label = gt[start_x:end_x + 1, start_y:end_y + 1, :]

                        # get input numpy
                        Iclean_crop = np.float32(label / 255.0)
                        # 1 * 512 * 512 * 3
                        Iclean_crop_exp = np.expand_dims(Iclean_crop, 0)
                        # 1 * 3 * 512 * 512
                        input = np.transpose(Iclean_crop_exp, (0, 3, 1, 2))

                        n = np.random.randint(0, args.ISP_num)
                        print("PseudoISP No. : %d\n" % (n + 1))

                        RGB2PACK_model = nn.DataParallel(RGB2PACK()).cuda()
                        PACK2RGB_model = nn.DataParallel(PACK2RGB()).cuda()
                        Noise_Model_Network_model = nn.DataParallel(Noise_Model_Network()).cuda()
                        load_model_dir = args.PseudoISP_path + '%04d_PseudoISP.pth' % (n + 1)
                        tmp_ckpt = torch.load(load_model_dir)

                        # RGB2PACK_model
                        pretrained_dict = tmp_ckpt['state_dict_RGB2PACK']
                        model_dict = RGB2PACK_model.state_dict()
                        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        assert (len(pretrained_dict) == len(pretrained_dict_update))
                        assert (len(pretrained_dict_update) == len(model_dict))
                        model_dict.update(pretrained_dict_update)
                        RGB2PACK_model.load_state_dict(model_dict)

                        # PACK2RGB_model
                        pretrained_dict = tmp_ckpt['state_dict_PACK2RGB']
                        model_dict = PACK2RGB_model.state_dict()
                        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        assert (len(pretrained_dict) == len(pretrained_dict_update))
                        assert (len(pretrained_dict_update) == len(model_dict))
                        model_dict.update(pretrained_dict_update)
                        PACK2RGB_model.load_state_dict(model_dict)

                        # Noise_Model_Network_model
                        pretrained_dict = tmp_ckpt['state_dict_Noise_Model_Network_model']
                        model_dict = Noise_Model_Network_model.state_dict()
                        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        assert (len(pretrained_dict) == len(pretrained_dict_update))
                        assert (len(pretrained_dict_update) == len(model_dict))
                        model_dict.update(pretrained_dict_update)
                        Noise_Model_Network_model.load_state_dict(model_dict)

                        with torch.no_grad():
                            RGB2PACK_model.eval()
                            PACK2RGB_model.eval()
                            Noise_Model_Network_model.eval()
                            # 1 * 3 * 512 * 512
                            input = torch.Tensor(input)
                            blur, pad = get_gaussian_kernel(kernel_size=5, sigma=1)
                            input = F.pad(input, (pad, pad, pad, pad), mode='reflect')
                            X_rgb = blur(input)
                            # 1 * 3 * 512 * 512
                            X_rgb = X_rgb.cuda()
                            X_pack, Y_pack = RGB2PACK_model(X_rgb, X_rgb)
                            Output = Noise_Model_Network_model(X_pack)

                            gamma_int = random.randint(90, 111)
                            # 0.9 ~ 1.1
                            gamma = gamma_int / 100.0
                            noise_level = Output * gamma

                            torch.manual_seed(0)
                            noise_map = torch.randn(X_pack.size()).cuda() * noise_level

                            noisy_image = X_pack + noise_map
                            # 1 * 3 * 512 * 512
                            X_output, Y_output = PACK2RGB_model(noisy_image, noisy_image)

                            img = Y_output.cpu().detach().clamp(0., 1.).numpy().astype(np.float32)
                            img = np.squeeze(img)
                            img = img.transpose(1, 2, 0)
                            img = np.uint8(np.round(img * 255.0))

                            img_gt = X_rgb.cpu().detach().clamp(0., 1.).numpy().astype(np.float32)
                            img_gt = np.squeeze(img_gt)
                            img_gt = img_gt.transpose(1, 2, 0)
                            img_gt = np.uint8(np.round(img_gt * 255.0))

                            count = 0
                            data_noisy = []
                            data_gt = []
                            stride = 120
                            for ii in range(10, img.shape[0] - patchsize - 10, stride):
                                for jj in range(10, img.shape[1] - patchsize -10, stride):
                                    x = img[ii:ii + patchsize, jj:jj + patchsize, :]
                                    y = img_gt[ii:ii + patchsize, jj:jj + patchsize, :]
                                    data_noisy.append(x)
                                    data_gt.append(y)
                                    count = count + 1
                            data_noisy = np.array(data_noisy)
                            data_gt = np.array(data_gt)
                            for nx in range(count):
                                input = data_noisy[nx, :, :, :].copy()
                                target = data_gt[nx, :, :, :].copy()
                                input = np.transpose(input, (2, 0, 1))
                                target = np.transpose(target, (2, 0, 1))
                                h5f.create_dataset(str(train_num), data=(input, target))
                                train_num += 1
# Pseudo Dataset Generation
def Pseudo_Paired_Dataset(args):
    train_num = 0
    patch_size, stride = 160, 60
    step1, step2 = 0, 0
    train_num = 0
    with h5py.File(args.traindbf_pre, 'w') as h5f:
        for n in range(50):
            for k in range(20):
                print("Cell No. : %d\n" % (k + 1))
                # 512 * 512 * 3
                noisy_valid_dir = args.dataset_path + args.Noisy_path + '%04d_%02d.mat' % (n + 1, k + 1)
                if not os.path.exists(noisy_valid_dir):
                    print("Please download dataset from my github")
                    print("The default path is " + args.dataset_path + args.Noisy_path)
                    print("You can change the path (args.dataset_path + args.Noisy_path)")
                    assert (os.path.exists(noisy_valid_dir))
                mat_file = sio.loadmat(noisy_valid_dir)
                # get input numpy
                Inoisy_crop = np.float32(np.array(mat_file['Inoisy_crop']))
                img = np.uint8(np.round(Inoisy_crop * 255.0))

                # 512 * 512 * 3
                denoised_valid_dir = args.dataset_path + args.Denoised_path + '%04d_%02d.mat' % (n + 1, k + 1)
                if not os.path.exists(denoised_valid_dir):
                    print("Please download denoised result from my github or your denoised result")
                    print("The default path is " + args.dataset_path + args.Denoised_path)
                    print("You can change the path ( args.dataset_path + args.Denoised_path)")
                    assert (os.path.exists(denoised_valid_dir))
                mat_file = sio.loadmat(denoised_valid_dir)
                Iclean_crop = np.float32(np.array(mat_file['Idenoised_crop']))
                # 512 * 512 * 3
                Iclean_crop = np.clip(Iclean_crop, 0., 1.)
                label = np.uint8(np.round(Iclean_crop * 255.0))
                count = 0
                data_noisy = []
                data_gt = []
                for ii in range(step1, img.shape[0] - patch_size + 1, stride):
                    for jj in range(step2, img.shape[1] - patch_size + 1, stride):
                        x = img[ii:ii + patch_size, jj:jj + patch_size, :]
                        y = label[ii:ii + patch_size, jj:jj + patch_size, :]
                        data_noisy.append(x)
                        data_gt.append(y)
                        count = count + 1
                print(str(count))
                data_noisy = np.array(data_noisy)
                data_gt = np.array(data_gt)
                for nx in range(count):
                    input = data_noisy[nx, :, :, :].copy()
                    target = data_gt[nx, :, :, :].copy()
                    input = np.transpose(input, (2, 0, 1))
                    target = np.transpose(target, (2, 0, 1))
                    h5f.create_dataset(str(train_num), data=(input, target))
                    train_num += 1
# We use our denoising results (PT-MWRN*) as a validation set
# You don't have to use it
def load_valid(args):
    # 20 * 3 * 512 * 512
    noisy_data = np.zeros((10, 3, 512, 512), dtype=np.float32)
    denoised_data = np.zeros((10, 3, 512, 512), dtype=np.float32)

    for i in range(10):
        if i == 0:
            n = 1
            k = 18
        if i == 1:
            n = 2
            k = 19
        if i == 2:
            n = 6
            k = 6
        if i == 3:
            n = 16
            k = 9
        if i == 4:
            n = 17
            k = 3
        if i == 5:
            n = 26
            k = 2
        if i == 6:
            n = 34
            k = 9
        if i == 7:
            n = 39
            k = 16
        if i == 8:
            n = 44
            k = 1
        if i == 9:
            n = 50
            k = 15

        noisy_valid_dir = args.datacroproot + 'noisy_part_mat/%04d_%02d.mat' % (n, k)
        mat_file = sio.loadmat(noisy_valid_dir)
        #  512 * 512 * 3
        Inoisy_crop = np.float32(np.array(mat_file['Inoisy_crop']))
        # 1 * 512 * 512 * 3
        Inoisy_crop_exp = np.expand_dims(Inoisy_crop, 0)
        # 1 * 3 * 512 * 512
        input = np.transpose(Inoisy_crop_exp, (0, 3, 1, 2))

        denoised_valid_dir = args.datacroproot + 'denoised_part_mat/%04d_%02d.mat' % (n, k)
        mat_file = sio.loadmat(denoised_valid_dir)
        #  512 * 512 * 3
        Iclean_crop = np.float32(np.array(mat_file['Idenoised_crop']))
        # 1 * 512 * 512 * 3
        Iclean_crop_exp = np.expand_dims(Iclean_crop, 0)
        # 1 * 3 * 512 * 512
        target = np.transpose(Iclean_crop_exp, (0, 3, 1, 2))

        noisy_data[i, :, :, :] = input
        denoised_data[i, :, :, :] = target
        # print(i)

    return noisy_data, denoised_data

def main(args):
    # with or without Pseudo paried as validation ?
    if args.valid_flag == 1:
        noisy_data, denoised_data = load_valid(args)
    if not os.path.exists('test_epoch_psnr.mat') and args.valid_flag == 1:
        s = {}
        s["tep"] = np.zeros((7, 1))
        sio.savemat('test_epoch_psnr.mat', s)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    initial_epoch = findLastCheckpoint(save_dir=args.save_path, save_pre = args.save_prefix)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        args.resume = "continue"
        args.last_ckpt = args.save_path + args.save_prefix + str(initial_epoch) + '.pth'

    # net architecture
    dn_net = Net()

    # loss function
    criterion = nn.MSELoss(reduction='sum').cuda()
    dn_model = nn.DataParallel(dn_net).cuda()
    # Optimizer
    training_params = None
    optimizer_dn = None
    # load old params, optimizer, state
    if args.resume == "continue":
        tmp_ckpt = torch.load(args.last_ckpt)
        start_epoch = initial_epoch + 1

        # Initialize dn_model
        pretrained_dict = tmp_ckpt['state_dict']
        model_dict=dn_model.state_dict()
        pretrained_dict_update = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert(len(pretrained_dict)==len(pretrained_dict_update))
        assert(len(pretrained_dict_update)==len(model_dict))
        model_dict.update(pretrained_dict_update)
        dn_model.load_state_dict(model_dict)
        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dn)
        optimizer_dn.load_state_dict(tmp_ckpt['optimizer_state'])
        schedule_dn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dn, milestones=args.steps, gamma=args.decay_rate)

    elif args.resume == "new":

        start_epoch = 1

        optimizer_dn = optim.Adam(dn_model.parameters(), lr=args.learning_rate_dn)
        schedule_dn = torch.optim.lr_scheduler.MultiStepLR(optimizer_dn, milestones=[20, 80], gamma=args.decay_rate)

    if args.resume=="continue" and args.valid_flag == 1:
        stg2_DN_valid.DN_valid(args, noisy_data.copy(), denoised_data.copy())


    # set training set DataLoader
    if not os.path.exists('./dataset_h5py/'):
        os.makedirs('./dataset_h5py/')
        # Synthesis Dataset Generation
        Generate_Synthesis_Dataset(args)
        # Pseudo Dataset Generation
        Pseudo_Paired_Dataset(args)
    dataset_train = Dataset(args.traindbf, shuffle=False, close_everytime=False, aug_mode=True)
    loader_train = DataLoader(dataset=dataset_train, num_workers=args.load_thread, batch_size=args.Synthesis_size,
                              shuffle=True, pin_memory=True, timeout=0)
    print("Number of Synthesis paried training images: %d\n" % int(len(dataset_train)))
    # Previous Dataset Generation
    if args.batch_size != args.Synthesis_size:
        dataset_previous = Dataset(args.traindbf_pre, shuffle=False, close_everytime=False, aug_mode=True)
        print("Number of Pseudo paried training images: %d\n" % int(len(dataset_previous)))
    total_step = len(loader_train)
    # training
    for epoch in range(start_epoch, args.epoch+1):
        # Synthesis Dataset Generation
        # if (epoch - start_epoch) % 30 == 0:
        schedule_dn.step()
        dn_model.train()
        # train
        tt = 0
        if args.batch_size != args.Synthesis_size:
            resultList = random.sample(range(0, int(len(dataset_previous))), int(len(dataset_previous)))
        i = 0
        for img_noise_1, img_train_1 in loader_train:

            batch, C, H, W = img_train_1.size()
            if batch == args.Synthesis_size:
                img_noise_1 = img_noise_1.cuda().float().div(255)
                img_train_1 = img_train_1.cuda().float().div(255)
                img_noise = torch.zeros(args.batch_size, C, H, W, device='cuda')
                img_train = torch.zeros(args.batch_size, C, H, W, device='cuda')

                bb_num = args.batch_size - args.Synthesis_size
                cc = 0
                for bb in range(bb_num):
                    tt = (tt + 1) % int(len(dataset_previous))
                    step_ram = resultList[tt]
                    img_noise_temp, img_train_temp = dataset_previous.__getitem__(step_ram)
                    img_noise[cc, :, :, :] = torch.unsqueeze(img_noise_temp, 0).cuda().float().div(255)
                    img_train[cc, :, :, :] = torch.unsqueeze(img_train_temp, 0).cuda().float().div(255)
                    cc = cc + 1

                # batch Simulation
                img_noise[cc:args.batch_size, :, :, :] = img_noise_1
                img_train[cc:args.batch_size, :, :, :] = img_train_1
                # 32 * 3 * 160 * 160
                optimizer_dn.zero_grad()


                output = dn_model(img_noise)

                loss = criterion(img_train, output)
                i = i + 1
                print("Epoch:[{}/{}] Batch: [{}/{}] loss = {:.4f}".format(epoch, args.epoch, i, total_step,
                                                                          loss.item() / batch))
                loss = loss / (2*batch)

                loss.backward()
                optimizer_dn.step()



        # save model and checkpoint

        save_dict = {'state_dict': dn_model.state_dict(),
                     'optimizer_state': optimizer_dn.state_dict(),
                     'schedule_state': schedule_dn.state_dict()}
        torch.save(save_dict, os.path.join(args.save_path + args.save_prefix + '{}.pth'.format(epoch)))
        del save_dict

        if epoch % args.save_every_epochs == 0 and args.valid_flag == 1:
            stg2_DN_valid.DN_valid(args, noisy_data.copy(), denoised_data.copy())





if __name__ == "__main__":

    main(opt)

    exit(0)




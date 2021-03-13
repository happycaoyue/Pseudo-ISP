# -*- coding: utf-8 -*-
"""
## Pseudo-ISP: Learning Pseudo In-camera Signal Processing Pipeline from A Color Image Denoiser
## Yue Cao, Xiaohe Wu, Shuran Qi, Xiao Liu, Zhongqin Wu, Wangmeng Zuo
## Thank the Professor Wangmeng Zuo for his guidance and help in this work.
## If you use our code, please cite our paper. Thank you.
## If you have a question or comment about our paper, please send me an email. cscaoyue@gamil.com
"""
import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from net.pseudoisp import RGB2PACK, PACK2RGB, Noise_Model_Network
import numpy as np
from stg1_PseudoISP_options import opt


class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss,self).__init__()
        self.L2 = nn.MSELoss(reduction='sum')
    def forward(self, X_rgb, Y_rgb, X_output, Y_output, X_pack, Y_pack, Output):
        # 1.253 = sqrt(3.1415926 / 2)
        loss = 0
        loss1 = self.L2(X_rgb, X_output)
        loss2 = self.L2(Y_rgb, Y_output)
        loss3 = self.L2(Output, (Y_pack-X_pack).abs() * 1.253)
        loss = loss1 + loss2 + 0.1 * loss3
        return loss

def main(args):

    print("********************Pseudo-ISP Experiment on DND dataset********************")
    # get input numpy
    num_imgs = 50
    print("DND Benchmark of images: %d\n" % (num_imgs))
    num_blocks = 20
    print("DND Benchmark of blocks/patches: %d\n" % (num_blocks))
    # 512 × 512 × 3
    H = 512
    W = 512
    C = 3
    print("Size of per blocks/patches: %d * %d * %d\n" % (H, W, C))
    psnr_all = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)

    # 1 ~ 50 images
    for n in range(num_imgs):

        print("DND Images No. : %d\n" % (n + 1))
        # 20 * 3 * 512 * 512
        noisy_data = np.zeros((num_blocks, C, H, W), dtype=np.float32)
        denoised_data = np.zeros((num_blocks, C, H, W), dtype=np.float32)

        # ./logs_DND_PseudoISP/
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        # ./logs_DND_PseudoISP/0001_img/
        log_dir = args.log_dir + '%04d_img/' % (n + 1)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # ./logs_DND_PseudoISP/0001_img/0001_model/
        save_path_model = log_dir + '%04d_model/' % (n + 1)
        if not os.path.exists(save_path_model):
            os.makedirs(save_path_model)
        # 1 ~ 20 blocks/patches
        for k in range(20):
            # 512 * 512 * 3
            noisy_valid_dir = args.datacroproot + 'noisy_mat/%04d_%02d.mat' % (n + 1, k + 1)
            if not os.path.exists(noisy_valid_dir):
                print("Please download dataset from my github")
                print("The default path is " + args.datacroproot + "noisy_mat/")
                print("You can change the path (args.datacroproot)")
                assert (os.path.exists(noisy_valid_dir))
            print(noisy_valid_dir)
            mat_file = sio.loadmat(noisy_valid_dir)
            # get input numpy
            Inoisy_crop = np.float32(np.array(mat_file['Inoisy_crop']))
            # 1 * 512 * 512 * 3
            Inoisy_crop_exp = np.expand_dims(Inoisy_crop, 0)
            # 1 * 3 * 512 * 512
            input = np.transpose(Inoisy_crop_exp, (0, 3, 1, 2))

            denoised_valid_dir = args.datacroproot + args.denoised_dir + '/%04d_%02d.mat' % (n + 1, k + 1)
            if not os.path.exists(denoised_valid_dir):
                print("Please download denoised result from my github or your denoised result")
                print("The default path is " + args.datacroproot + args.denoised_dir)
                print("You can change the path ( args.datacroproot + args.denoised_dir)")
                assert (os.path.exists(denoised_valid_dir))
            print(denoised_valid_dir)
            mat_file = sio.loadmat(denoised_valid_dir)
            # get input numpy
            Iclean_crop = np.float32(np.array(mat_file['Idenoised_crop']))
            Iclean_crop = np.clip(Iclean_crop, 0., 1.)
            # 1 * 512 * 512 * 3
            Iclean_crop_exp = np.expand_dims(Iclean_crop, 0)
            # 1 * 3 * 512 * 512
            target = np.transpose(Iclean_crop_exp, (0, 3, 1, 2))

            noisy_data[k, :, :, :] = input
            denoised_data[k, :, :, :] = target

        # loss function
        criterion = MultiLoss().cuda()

        RGB2PACK_model = nn.DataParallel(RGB2PACK()).cuda()
        PACK2RGB_model = nn.DataParallel(PACK2RGB()).cuda()
        Noise_Model_Network_model = nn.DataParallel(Noise_Model_Network()).cuda()

        optimizer_RGB2PACK = optim.Adam(RGB2PACK_model.parameters(), lr=args.learning_rate_RGB2PACK)
        optimizer_PACK2RGB = optim.Adam(PACK2RGB_model.parameters(), lr=args.learning_rate_PACK2RGB)
        optimizer_Noise_Model_Network = optim.Adam(Noise_Model_Network_model.parameters(), lr=args.learning_rate_Noise_Model_Network)

        schedule_RGB2PACK = torch.optim.lr_scheduler.MultiStepLR(optimizer_RGB2PACK, milestones=[8000, 12000], gamma=args.decay_rate)
        schedule_PACK2RGB = torch.optim.lr_scheduler.MultiStepLR(optimizer_PACK2RGB, milestones=[8000, 12000], gamma=args.decay_rate)
        schedule_Noise_Model_Network = torch.optim.lr_scheduler.MultiStepLR(optimizer_Noise_Model_Network, milestones=[8000, 12000], gamma=args.decay_rate)

        # original noisy image
        # numpy -> torch
        # 20 * 3 * 512 * 512
        img_noisy = torch.from_numpy(noisy_data).cuda()
        img_denoised = torch.from_numpy(denoised_data).cuda()

        RGB2PACK_model.train()
        PACK2RGB_model.train()
        Noise_Model_Network_model.train()
        max_epoch = 0
        for epoch in range(args.epoch):

            optimizer_RGB2PACK.zero_grad()
            optimizer_PACK2RGB.zero_grad()
            optimizer_Noise_Model_Network.zero_grad()

            # batchsize * C * patchsize * patchsize  32 * 3 * 60 * 60
            Y_rgb = torch.zeros(args.batch_size, C, args.patch_size, args.patch_size, device='cuda')
            X_rgb = torch.zeros(args.batch_size, C, args.patch_size, args.patch_size, device='cuda')

            for j in range(args.batch_size):
                num, idx1, idx2 = np.random.randint(num_blocks), np.random.randint(0, H - args.patch_size), np.random.randint(0, W - args.patch_size)
                Y_rgb[j, :, :, :] = img_noisy[num, :, idx1:idx1 + args.patch_size, idx2:idx2 + args.patch_size]
                X_rgb[j, :, :, :] = img_denoised[num, :, idx1:idx1 + args.patch_size, idx2:idx2 + args.patch_size]

            # 32 * 3 * 60 * 60 -> 32 * 4 * 30 * 30
            X_pack, Y_pack = RGB2PACK_model(X_rgb, Y_rgb)
            # 32 * 4 * 30 * 30 -> 32 * 3 * 60 * 60
            X_output, Y_output = PACK2RGB_model(X_pack, Y_pack)
            # 32 * 4 * 30 * 30 -> 32 * 4 * 30 * 30
            Output = Noise_Model_Network_model(X_pack)

            loss = criterion(X_rgb, Y_rgb, X_output, Y_output, X_pack, Y_pack, Output)

            loss = loss / (args.batch_size)
            loss.backward()
            optimizer_RGB2PACK.step()
            optimizer_PACK2RGB.step()
            optimizer_Noise_Model_Network.step()
            loss_value = loss.item()
            print("Epoch:[{}/{}] loss = {:.4f}".format(epoch, args.epoch, loss_value))

            schedule_RGB2PACK.step(epoch)
            schedule_PACK2RGB.step(epoch)
            schedule_Noise_Model_Network.step(epoch)


            if (epoch+1) % args.save_model_freq == 0:


                # ./logs_DND_PseudoISP/0001_img/0001_model/0001_e01000_model/
                save_path_epoch_model = save_path_model + '%04d_e%05d_model/' % (n + 1, epoch + 1)
                if not os.path.exists(save_path_epoch_model):
                    os.makedirs(save_path_epoch_model)


                save_dict = {'state_dict_RGB2PACK': RGB2PACK_model.state_dict(),
                            'optimizer_RGB2PACK_state': optimizer_RGB2PACK.state_dict(),
                            'state_dict_PACK2RGB': PACK2RGB_model.state_dict(),
                            ' optimizer_PACK2RGB_state': optimizer_PACK2RGB.state_dict(),
                             'state_dict_Noise_Model_Network_model': Noise_Model_Network_model.state_dict(),
                             'optimizer_Noise_Model_Network_state': optimizer_Noise_Model_Network.state_dict()}

                # ./logs_DND_PseudoISP/0001_img/0001_model/0001_e01000_model/0001_PseudoISP.pth
                save_model_file = save_path_epoch_model + '%04d_PseudoISP.pth' % (n + 1)
                torch.save(save_dict, save_model_file)
                del save_dict
                # 20 * 3 * 512 * 512
                max_epoch = epoch + 1

        # model
        # ./logs_DND_PseudoISP/0001_img/0001_model/0001_e01000_model/0001_PseudoISP.pth
        load_model_dir = save_path_model + '%04d_e%05d_model/%04d_PseudoISP.pth' % (n + 1, max_epoch, n + 1)
        temp_model = torch.load(load_model_dir)
        if not os.path.exists('./PseudoISP_ckpt/'):
            os.makedirs('./PseudoISP_ckpt/')
        save_model_dir = './PseudoISP_ckpt/%04d_PseudoISP.pth' % (n + 1)
        torch.save(temp_model, save_model_dir)


if __name__ == "__main__":

    main(opt)

    exit(0)




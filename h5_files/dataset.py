import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
def modcrop(image, scale):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image
def normalize(data):
	r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
	return np.float32(data/255.)

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
	out = np.transpose(image, (1, 2, 0))
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
	return np.transpose(out, (2, 0, 1))

def img2patches_mask_pixel(img, win):
    endc = img.shape[0]
    endh = img.shape[1]
    endw = img.shape[2]
    msk=torch.ones((win,win))
    msk[win//2,win//2]=0
    win2=win//2
    Mask=torch.zeros((endh+win2*2,endw+win2*2))
    Mask[win2:-win2,win2:-win2]=torch.ones((endh,endw))
    pad=torch.zeros((endc,endh+win2*2,endw+win2*2))
    pad[:,win2:-win2,win2:-win2]=img
    res=torch.zeros((endh,endw,endc,win,win))
    masks=torch.zeros((endh,endw,win,win))
    for i in range(endh):
        for j in range(endw):
            res[i,j,:,:,:]=pad[:,i:i+win,j:j+win]
            masks[i,j,:,:]=msk*Mask[i:i+win,j:j+win]
    res=res.view(-1,endc,win,win)
    masks=masks.view(-1,1,win,win)
    pixel=img.view(-1)
    return res,masks,pixel

def img_to_patches(img, win, stride=1):
    r"""Converts an image to an array of patches.

    Args:
        img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
            image
        win: size of the output patches
        stride: int. stride
    """
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    if endw<win or endh<win:
        return np.zeros([endc,win,win,0])
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    res = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return res.reshape([endc, win, win, total_pat_num])
# def prepare_data(data_path, val_data_path, patch_size, stride, scales = [1, 0.9, 0.8, 0.7],
def prepare_data(data_path, val_data_path, patch_size, stride, scales = [1],
                 max_num_patches=None, aug_times=1,random_aug=False, gray_mode=False):
    r"""Builds the training and validations datasets by scanning the
    corresponding directories for images and extracting	patches from them.

    Args:
        data_path: path containing the training image dataset
        val_data_path: path containing the validation image dataset
        patch_size: size of the patches to extract from the images
        stride: size of stride to extract patches
        stride: size of stride to extract patches
        max_num_patches: maximum number of patches to extract
        aug_times: number of times to augment the available data minus one
        gray_mode: build the databases composed of grayscale patches
    """
    # training database
    print('> Training database')
    types = ('*.jpg','*.bmp', '*.png', '*.tif')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp)))
    files.sort()

    if gray_mode:
        traindbf = 'trainset_p40_s10_gray.h5'
        # traindbf = 'trainset_gray.h5'
        # traindbf = '/share/data/cy/trainset_p80_s160_gray.h5'
        valdbf = 'BSD68.h5'

    else:
        traindbf = 'trainset_p128_s200_rgb.h5'
        valdbf = 'Kodak24_color.h5'
        # valdbf = 'set12_color.h5'
    if max_num_patches is None:
        max_num_patches = 2000000
        #max_num_patches = 300
        print("\tMaximum number of patches not set")
    else:
        print("\tMaximum number of patches set to {}".format(max_num_patches))
    train_num = 0
    i = 0
    with h5py.File(traindbf, 'w') as h5f:
        while i < len(files) and train_num < max_num_patches:
            imgor = cv2.imread(files[i])
            # h, w, c = img.shape
            for sca in scales:
                img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, \
                                 interpolation=cv2.INTER_CUBIC)
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
                img = normalize(img)
                patches = img_to_patches(img, win=patch_size, stride=stride)
                print("\tfile: %s scale %.1f # samples: %d" % \
                      (files[i], sca, patches.shape[3]*8))
                for nx in range(patches.shape[3]):
                    if random_aug==False:
                        for j in range(aug_times):
                            data = data_augmentation(patches[:, :, :, nx].copy(),j)
                            h5f.create_dataset(str(train_num), data=data)
                            train_num += 1
                    else:
                        for j in range(aug_times):
                            data = data_augmentation(patches[:, :, :, nx].copy(), random.randint(0,7))
                            h5f.create_dataset(str(train_num), data=data)
                            train_num += 1
            i += 1
    
    val_num = 0
    # validation database
    print('\n> Validation database')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in enumerate(files):
        print("\tfile: %s" % item)
        img = cv2.imread(item)
        img = modcrop(img, 8)
        if not gray_mode:
            # C. H. W, RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
        else:
            # C, H, W grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)

        C,H,W=img.shape

        # if H % 2 == 1:
        # 	img = img[:, :-1, :]
        # if W % 2 == 1:
        # 	img = img[:, :, :-1]

        img = normalize(img)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()

    print('\n> Total')
    print('\ttraining set, # samples %d' % train_num)
    print('\tvalidation set, # samples %d\n' % val_num)

if __name__ == "__main__":
    # 数据集
    # 训练集合
    data_path="/home/sr/桌面/TCN_new/h5_files/DIV2K_train_HR/"
    # data_path = "/home/sr/桌面/TCN_new/h5_files/Train400/"
    # data_path = "D:/dataset/DIV2K_train_HR/"
    # data_path="/share/Dataset/DIV2K_train_HR/"
    # data_path="./BSD68/"

    # data_path = "/home/sr/桌面/TCN_new/h5_files/CBSD432/"
    # data_path = "/home/sr/桌面/TCN_new/h5_files/CBSD1/"

    # 测试集合
    # val_data_path = "./BSD68/"
    val_data_path = "/home/sr/桌面/TCN_new/h5_files/Kodak24/"
    patch_size = 128
    stride = 200
    prepare_data(data_path=data_path, val_data_path=val_data_path, patch_size=patch_size, stride=stride)
    exit(0)

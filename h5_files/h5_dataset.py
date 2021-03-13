import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from util.utils import data_augmentation, normalize


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

def prepare_data(data_path, val_data_path, patch_size,stride,scales = [1, 0.9, 0.8, 0.7],
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
    types = ('*.bmp', '*.png')
    files = []
    for tp in types:
        files.extend(glob.glob(os.path.join(data_path, tp)))
    files.sort()

    if gray_mode:
        traindbf = './data/set400_p64.h5'
        valdbf = './data/set12.h5'
    else:
        traindbf = './data/train_rgb.h5'
        valdbf = './data/val_rgb.h5'

    if max_num_patches is None:
        max_num_patches = 5000000
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
                      (files[i], sca, patches.shape[3] * 8))
                for nx in range(patches.shape[3]):
                    if random_aug == False:
                        for j in range(aug_times):
                            data = data_augmentation(patches[:, :, :, nx].copy(), j)
                            h5f.create_dataset(str(train_num), data=data)
                            train_num += 1
                    else:
                        for j in range(aug_times):
                            data = data_augmentation(patches[:, :, :, nx].copy(), random.randint(0, 7))
                            h5f.create_dataset(str(train_num), data=data)
                            train_num += 1
            i += 1
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
class Dataset(udata.Dataset):
    r"""Implements torch.utils.data.Dataset
    """
    def __init__(self, file_name, shuffle=False,close_everytime=True, aug_mode = False):
        super(Dataset, self).__init__()
        if not os.path.exists(file_name):
            raise("file_name is not valid")
        self.aug_mode = aug_mode
        self.close_everytime=close_everytime
        self.file_name=file_name
        if self.close_everytime:
            h5f = h5py.File(self.file_name, 'r')
            self.keys = list(h5f.keys())
            if shuffle:
                random.shuffle(self.keys)
            h5f.close()
        else:
            self.h5f = h5py.File(self.file_name, 'r')
            self.keys = list(self.h5f.keys())
            self.img_dict=dict()
            for key in self.keys:
                self.img_dict[key]=np.array(self.h5f[key])
                # self.img_dict[key]=torch.FloatTensor(self.h5f[key])
            if shuffle:
                random.shuffle(self.keys)
            self.h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.close_everytime:
            h5f = h5py.File(self.file_name, 'r')
            key = self.keys[index]
            #data = np.array(h5f[key])
            (noisy, clean) = h5f[key]
            if self.aug_mode:
                mode_index = np.random.randint(0, 8)
                noisy = np.array(noisy)
                clean = np.array(clean)
                # noisy = noisy.astype(np.float32)
                # clean = clean.astype(np.float32)
                noisy = data_augmentation(noisy, mode_index)
                clean = data_augmentation(clean, mode_index)
                noisy = torch.Tensor(noisy.copy())
                clean = torch.Tensor(clean.copy())
            else:
                noisy = np.array(noisy)
                clean = np.array(clean)
                # noisy = noisy.astype(np.float32)
                # clean = clean.astype(np.float32)
                noisy = torch.Tensor(noisy.copy())
                clean = torch.Tensor(clean.copy())

            h5f.close()
        else:
            key=self.keys[index]
            #data=np.array(self.img_dict[key])
            (noisy, clean)=self.img_dict[key]
            if self.aug_mode:
                mode_index = np.random.randint(0, 8)
                noisy = np.array(noisy)
                clean = np.array(clean)
                noisy = data_augmentation(noisy, mode_index)
                clean = data_augmentation(clean, mode_index)
                noisy = torch.Tensor(noisy.copy())
                clean = torch.Tensor(clean.copy())
            else:
                noisy = torch.Tensor(noisy)
                clean = torch.Tensor(clean)
        return noisy, clean
    
"""
class Dataset(udata.Dataset):
    # Implements torch.utils.data.Dataset
    def __init__(self, file_name, shuffle=False,close_everytime=True):
        super(Dataset, self).__init__()
        if not os.path.exists(file_name):
            raise("file_name is not valid")
        self.close_everytime=close_everytime
        self.file_name=file_name
        if self.close_everytime:
            h5f = h5py.File(self.file_name, 'r')
            self.keys = list(h5f.keys())
            if shuffle:
                random.shuffle(self.keys)
            h5f.close()
        else:
            self.h5f = h5py.File(self.file_name, 'r')
            self.keys = list(self.h5f.keys())
            self.img_dict=dict()
            for key in self.keys:
                self.img_dict[key]=np.array(self.h5f[key])
            if shuffle:
                random.shuffle(self.keys)
            self.h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.close_everytime:
            h5f = h5py.File(self.file_name, 'r')
            key = self.keys[index]
            data = np.array(h5f[key])
            h5f.close()
        else:
            key=self.keys[index]
            data=np.array(self.img_dict[key])
        return torch.Tensor(data)
"""

class DatasetTrain(udata.Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, data_paths,patch_size,stride,shuffle=True,gray_mode=True,contrast=False,random_aug=False,aug_times=1):
        super(DatasetTrain, self).__init__()
        self.gray_mode=gray_mode
        self.contrast=contrast
        if contrast==False:
            self.data_list=list()
            for data_path in data_paths:
                self.data_list.extend(self.datagenerator(data_path,patch_size,stride,gray_mode=gray_mode,random_aug=random_aug,aug_times=aug_times))
            if shuffle:
                random.shuffle(self.data_list)
        else:
            assert len(data_paths)==2,'the len of data_paths must be 2,since Contrast is True'
            self.data_list_a=self.datagenerator(data_paths[0],patch_size,stride,gray_mode=gray_mode,random_aug=False,aug_times=1)
            self.data_list_b=self.datagenerator(data_paths[1],patch_size,stride,gray_mode=gray_mode,random_aug=False,aug_times=1)
            assert len(self.data_list_a)==len(self.data_list_b),'number of contrast patches must match'
            self.data_list=list(zip(self.data_list_a,self.data_list_b))
            if shuffle:
                random.shuffle(self.data_list)


    def datagenerator(self,data_path,patch_size,stride,gray_mode=True,scales = [1, 0.9, 0.8, 0.7],
                      max_num_patches=None, aug_times=1,random_aug=False):
        types = ('*.bmp', '*.png','*.jpg','*.jpeg')
        files = []
        for tp in types:
            files.extend(glob.glob(os.path.join(data_path, tp)))
        if max_num_patches is None:
            max_num_patches = 5000000
            print("\tMaximum number of patches not set")
        else:
            print("\tMaximum number of patches set to {}".format(max_num_patches))
        train_num = 0
        i = 0
        data_list=list()
        while i < len(files) and train_num < max_num_patches:
            imgor = cv2.imread(files[i])
            # h, w, c = img.shape
            for sca in scales:
                img = cv2.resize(imgor, (0, 0), fx=sca, fy=sca, interpolation=cv2.INTER_CUBIC)
                if not gray_mode:
                    # CxHxW RGB image
                    img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
                else:
                    # CxHxW grayscale image (C=1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.expand_dims(img, 0)
                img = normalize(img)
                patches = img_to_patches(img, win=patch_size, stride=stride)
                for nx in range(patches.shape[3]):
                    if random_aug==False:
                        for j in range(aug_times):
                            data = data_augmentation(patches[:, :, :, nx].copy(), j)
                            data_list.append(data)
                            train_num+=1
                    else:
                        data=data_augmentation(patches[:, :, :, nx].copy(),random.randint(0,7))
                        data_list.append(data)
                        train_num+=1
            i += 1
        print('training set, # samples %d' % train_num)
        return data_list

    def __getitem__(self, index):
        sample = self.data_list[index]
        if not self.contrast:
            return sample
        else:
            sample_a,sample_b=sample
            return sample_a,sample_b

    def __len__(self):
        return len(self.data_list)

class DatasetVal(udata.Dataset):
    def __init__(self,data_paths,gray_mode=True,contrast=False):
        self.gray_mode=gray_mode
        self.contrast=contrast
        if contrast==False:
            self.data_list=list()
            for data_path in data_paths:
                self.data_list.extend(self.datalist(data_path,gray_mode=gray_mode))
        else:
            assert len(data_paths)==2,'the len of data_paths must be 2,since Contrast is True'
            self.data_list_a=self.datalist(data_paths[0],gray_mode=gray_mode)
            self.data_list_b=self.datalist(data_paths[1],gray_mode=gray_mode)
            assert len(self.data_list_a)==len(self.data_list_b),'number of contrast patches must match'
            self.data_list=list(zip(self.data_list_a,self.data_list_b))

    def datalist(self,data_path,gray_mode=True):
        types = ('*.bmp', '*.png', '*.jpg', '*.jpeg')
        files = []
        for tp in types:
            files.extend(glob.glob(os.path.join(data_path, tp)))
        data_list = list()
        i=0
        while i < len(files):
            img = cv2.imread(files[i])
            # h, w, c = img.shape
            if not gray_mode:
                # CxHxW RGB image
                img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
            else:
                # CxHxW grayscale image (C=1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, 0)
            img = normalize(img)
            data_list.append(img)
            i += 1
        return data_list

    def __getitem__(self, index):
        sample = self.data_list[index]
        if not self.contrast:
            return sample
        else:
            sample_a,sample_b=sample
            return sample_a,sample_b

    def __len__(self):
        return len(self.data_list)
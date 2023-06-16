# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import random
import os
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
import scipy.misc as misc



class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image = (image-image.min())/(image.max()-image.min())
        mask = mask/255.0
        if mask is None:
            return image
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask=None):
        H,W   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1].copy(), mask[:,::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask

def Resize(image, mask,H,W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask  = cv2.resize( mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        return image, mask
    else:
        return image

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)

        return image, mask

def _resize_image(image, target):
   return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_LINEAR)


#root = '/home/ziyun/Desktop/Project/Mice_seg/Data_train'
class MyDataset(data.Dataset):# 
    def __init__(self, train_root, mode='train'): 
        img_path = '/train/original_images/'
        gt_path = '/train/masks/'

        img_ls = []
        mask_ls = []
        name_ls = []

        for pat_id in train_root:

            img_l = glob.glob(img_path+pat_id+'*')

            for img in img_l:
                gt = img.split('original_images/')[0]+'masks/'+img.split('original_images/')[1]
                name = img.split('original_images/')[1].split('.png')[0]
                img_ls.append(img)
                mask_ls.append(gt)
                name_ls.append(name)

        self.mode = mode
        self.name_ls = name_ls
        self.img_ls = img_ls

        self.mask_ls = mask_ls

        self.normalize  = Normalize()
        self.randomflip = RandomFlip()

        self.totensor   = ToTensor()

    def __getitem__(self, index):

        img  = cv2.imread(self.img_ls[index],0).astype(np.float32)
        mask  = cv2.imread(self.mask_ls[index], 0).astype(np.float32)
        # print(img.max(),mask.max())

        if self.mode == 'train':
            img, mask = self.normalize(img, mask)
            img, mask = Resize(img, mask,256,256)
            # 
            img, mask = self.randomflip(img, mask)
            img, mask = self.totensor(img, mask)
            mask = torch.where(mask>0.5,1,0).float()
            img, mask = img.unsqueeze(0), mask.unsqueeze(0)
            conn = connectivity_matrix(mask)
            return img,mask,conn
        else:
            img, mask = self.normalize(img, mask)
            img, mask = Resize(img, mask,256,256)

            img, mask = self.totensor(img, mask)
            mask = torch.where(mask>0.5,1,0).float()
            img, mask = img.unsqueeze(0), mask.unsqueeze(0)
            name = self.name_ls[index]
            return img, mask,name



    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.img_ls)


def connectivity_matrix(mask):
    # print(mask.shape)
    [channels,rows, cols] = mask.shape

    conn = torch.ones([8,rows, cols])
    up = torch.zeros([rows, cols])#move the orignal mask to up
    down = torch.zeros([rows, cols])
    left = torch.zeros([rows, cols])
    right = torch.zeros([rows, cols])
    up_left = torch.zeros([rows, cols])
    up_right = torch.zeros([rows, cols])
    down_left = torch.zeros([rows, cols])
    down_right = torch.zeros([rows, cols])


    up[:rows-1, :] = mask[0,1:rows,:]
    down[1:rows,:] = mask[0,0:rows-1,:]
    left[:,:cols-1] = mask[0,:,1:cols]
    right[:,1:cols] = mask[0,:,:cols-1]
    up_left[0:rows-1,0:cols-1] = mask[0,1:rows,1:cols]
    up_right[0:rows-1,1:cols] = mask[0,1:rows,0:cols-1]
    down_left[1:rows,0:cols-1] = mask[0,0:rows-1,1:cols]
    down_right[1:rows,1:cols] = mask[0,0:rows-1,0:cols-1]

    # print(mask.shape,down_right.shape)
    conn[0] = mask[0]*down_right
    conn[1] = mask[0]*down
    conn[2] = mask[0]*down_left
    conn[3] = mask[0]*right
    conn[4] = mask[0]*left
    conn[5] = mask[0]*up_right
    conn[6] = mask[0]*up
    conn[7] = mask[0]*up_left
    conn = conn.float()

    return conn



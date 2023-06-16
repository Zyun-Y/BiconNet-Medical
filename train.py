import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from GetDataset import MyDataset
from model.BaseNet import CPFNet
import glob
from torchvision import datasets, transforms
from model.unet_model import UNet
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn
from GetDataset import MyDataset, connectivity_matrix
# from GetDataset_test import MyDataset_test
from model.cenet import CE_Net_
import cv2
from skimage.io import imread, imsave
#torch.set_default_tensor_type('torch.FloatTensor')
# import minerl
# import gym
torch.cuda.set_device(0)
# gym.logger.set_level(40)


def main():
    overall_id = ['patient1','patient2','patient3','patient4']
    total_pat = [0,1,2,4]
    K_fold = 3

    ### the for loop here is for K-fold validation. You can remove it if you don't need.
    for exp_id in range(K_fold):
        ### define your fold and load your data from this part###
        test_id = [overall_id[exp_id]]
        train_id = list(set(total_pat)-set([exp_id]))
        train_id = [overall_id[i] for i in train_id]

        train_data = MyDataset(train_root = train_id,mode='train')
        test_data = MyDataset(train_root = test_id,mode='test')
        print("Train size: %i" % len(train_data))
        print("Test size: %i" % len(test_data))


        train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=8, shuffle=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=1, shuffle=False, num_workers=4)

        class_num = 1
        ### you can use any medical model here. E.g., U-Net, CE-Net.
        ### Please change the output channel to 8*N, if you have N classes. 
        model = CE_Net_(num_classes=8*class_num, num_channels=3).cuda()


        solver = Solver(lr= 0.0006)

        solver.train(model, train_loader, val_loader,exp_id+1,num_epochs=80)

if __name__ == '__main__':
    main()
    

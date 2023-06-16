import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.io as scio

Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
TL_table = ['lower_right','down','lower_left','right','left','upper_right','up','upper_left']



class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def soft_dice_coeff(self, y_pred,y_true):
        smooth = 0.001  # may change

        i = torch.sum(y_true,dim=(1,2))
        j = torch.sum(y_pred,dim=(1,2))
        intersection = torch.sum(y_true * y_pred,dim=(1,2))


        score = (2. * intersection + smooth) / (i + j + smooth)

        return (1-score),i
    def soft_dice_loss(self, y_pred,y_true):
        loss, gt_cnt = self.soft_dice_coeff(y_true, y_pred)

        return loss.mean()

    def __call__(self, y_pred,y_true):

        b = self.soft_dice_loss(y_true, y_pred)
        return b




def edge_loss(vote_out,con_target):
    # print(vote_out.shape,con_target.shape)
    sum_conn = torch.sum(con_target.clone(),dim=1)
    edge = torch.where((sum_conn<8) & (sum_conn>0),torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))

    pred_mask_min, _ = torch.min(vote_out.cuda(), dim=1)

    pred_mask_min = pred_mask_min*edge

    minloss = F.binary_cross_entropy(pred_mask_min,torch.full_like(pred_mask_min, 0))
    # print(minloss)
    return minloss#+maxloss

class connect_loss(nn.Module):
    def __init__(self,class_num=1):
        super(connect_loss, self).__init__()
        self.BCEloss = nn.BCELoss()
        self.dice_loss = dice_loss()
        self.class_num = class_num

    def forward(self, c_map, target, con_target,hori_translation,verti_translation):

        # print(c_map.shape,target.shape,con_target.shape,hori_translation.shape)
        con_target = con_target.type(torch.FloatTensor).cuda()
        target = target.type(torch.FloatTensor).cuda()

        hori_translation = hori_translation.repeat(c_map.shape[0],1,1,1).cuda()
        verti_translation = verti_translation.repeat(c_map.shape[0],1,1,1).cuda()

        self.hori_translation = hori_translation.cuda()
        self.verti_translation = verti_translation.cuda()



        c_map = F.sigmoid(c_map)

        final_pred, bimap = self.ConMap2Mask_prob(c_map)
        
        decouple_loss = edge_loss(bimap.squeeze(1),con_target)
        loss_dice = self.dice_loss(final_pred, target)
        bce_loss = self.BCEloss(final_pred,target)
        conn_l = self.BCEloss(c_map,con_target)
        bicon_l = self.BCEloss(bimap.squeeze(),con_target)
        loss =  bce_loss + conn_l+loss_dice  + 0.2* bicon_l + decouple_loss#+loss_dice #+ bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice

        return loss

    def shift_diag(self,img,shift):
        ## shift = [1,1] moving right and down
        # print(img.shape,self.hori_translation.shape)
        batch,class_num, row, column = img.size()

        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),self.hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),self.hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(self.verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(self.verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch,class_num, row, column)


    def ConMap2Mask_prob(self,c_map):
        c_map = c_map.view(c_map.shape[0],self.class_num,8,c_map.shape[2],c_map.shape[3])
        batch,class_num,channel, row, column = c_map.size()

        # print(c_map.shape)
        shifted_c_map = torch.zeros(c_map.size()).cuda()
        for i in range(8):
            shifted_c_map[:,:,i] = self.shift_diag(c_map[:,:,7-i].clone(),Directable[TL_table[i]])
        vote_out = c_map*shifted_c_map

        pred_mask,_ = torch.max(vote_out,dim=2)
        # print(pred_mask)
        return pred_mask,vote_out#, bimap


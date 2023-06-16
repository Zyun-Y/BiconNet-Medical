from random import shuffle
import numpy as np
from connect_loss import connect_loss
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from lr_update import get_lr
from torch.optim import lr_scheduler

import os
from cldice import clDice
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from skimage.io import imread, imsave

Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
TL_table = ['lower_right','down','lower_left','right','left','upper_right','up','upper_left']

save = 'save'
if not os.path.isdir(save):
    os.makedirs(save)


def create_exp_directory(exp_id):
    if not os.path.exists('models/'+str(exp_id)):
        os.makedirs('models/'+str(exp_id))


class Solver(object):
    # global optimiser parameters
    default_optim_args = {"lr": 1e-2,
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "weight_decay": 0.0001}


    def __init__(self, optim=torch.optim.Adam, lr=0.001,
                class_num=1):
        self.lr = lr
        self.optim = optim
        self.loss_func = connect_loss(class_num)

        self.class_num = class_num

        self.hori_trans = torch.zeros([1,self.class_num,256,256]).cuda()
        for i in range(255):
            self.hori_trans[:,:,i,i+1] = torch.tensor(1.0)
        self.verti_trans = torch.zeros([1,self.class_num,256,256]).cuda()
        for j in range(255):
            self.verti_trans[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_trans = self.hori_trans.float()
        self.verti_trans = self.verti_trans.float()



    def train(self, model, train_loader, val_loader, exp_id=0,num_epochs=10):
        
        optim = self.optim(model.parameters(), lr =self.lr)
        # scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,
        #                                 gamma=self.gamma)  # decay LR by a factor of 0.5 every 5 epochs  

        model.cuda()

        print('START TRAIN.')
        
        create_exp_directory(exp_id)
        
        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, dice, Jac \n')

        

 
        best_p = 0

        best_epo = 0
        # self.test_epoch(model,val_loader,1,exp_id)
        
        for epoch in range(num_epochs):

            model.train()
            # scheduler.step()
            curr_lr = get_lr(self.lr,'poly', epoch, num_epochs)

            for param_group in optim.param_groups:
                param_group['lr'] = curr_lr
            print(param_group['lr'])

            for i_batch, sample_batched in enumerate(train_loader):
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])
                conn = Variable(sample_batched[2])

                X= X.cuda()

                y = y.cuda()
                conn = conn.cuda()

                optim.zero_grad()

                s_output = model(X)

                
                loss = self.loss_func(s_output, y,conn,self.hori_trans,self.verti_trans)


                loss.backward()
                optim.step()
                
                
                print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f ' %(
                    loss.item()))


            dice_p = self.test_epoch(model,val_loader,epoch,exp_id)
            
            if best_p<dice_p:
                best_p = dice_p
                best_epo = epoch
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_model.pth')
            # if epoch%5==0:
            #     torch.save(model.state_dict(), 'models/' + str(exp_id) + '/relaynet_epoch' + str(epoch + 1)+'.pth')
        csv = 'results.csv'
        with open(os.path.join(save, csv), 'a') as f:
            f.write('%03d,%0.6f \n' % (
                best_epo,
                best_p
            ))


        print('FINISH.')
        
    def test_epoch(self,model,loader,epoch,exp_id):

        model.eval()
        # cnt = 0
        self.dice_ls = []
        self.Jac_ls=[]
        cldc_ls=[]
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                name = test_data[2]
                
                X_test= X_test.cuda()

                y_test = y_test.cuda()

                output_test = model(X_test)
                self.hori_translation = self.hori_trans.repeat(X_test.shape[0],1,1,1)
                self.verti_translation = self.verti_trans.repeat(X_test.shape[0],1,1,1)


                output_test = F.sigmoid(output_test)
                pred = torch.where(output_test>0.5,1,0)

                pred,_ = self.ConMap2Mask_prob(pred)


                dice,Jac = self.per_class_dice(pred,y_test)
                
                pred_np = pred.squeeze().cpu().numpy()
                target_np = y_test.squeeze().cpu().numpy()
                cldc = clDice(pred_np,target_np)
                cldc_ls.append(cldc)
                self.dice_ls += dice.tolist()
                self.Jac_ls += Jac.tolist()

                if j_batch%20==0:
                    print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                        np.mean(self.dice_ls)))

            Jac_ls =np.array(self.Jac_ls)
            dice_ls = np.array(self.dice_ls)
            total_dice = np.mean(dice_ls)
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(save, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch + 1),
                    total_dice,
                    np.mean(Jac_ls),
                    np.mean(cldc_ls)
                    # np.mean(self.acc_to)
                ))


            return np.mean(self.dice_ls)

    def per_class_dice(self,y_pred, y_true):

        FN = torch.sum((1-y_pred)*y_true,dim=(1,2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(1,2,3)) 

        inter = torch.sum(y_true* y_pred,dim=(1,2,3)) 

        union = torch.sum(y_true,dim=(1,2,3)) + torch.sum(y_pred,dim=(1,2,3)) 
        Jac = inter/(inter+FP+FN+0.0001)

        return (2*inter+0.0001)/(union+0.0001),Jac


    def shift_diag(self,img,shift):
        ## shift = [1,1] moving right and down
        # print(img.shape,self.hori_translation.shape)
        batch,class_num, row, column = img.size()
        img = img.float()
        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),self.hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),self.hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(self.verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(self.verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch,class_num, row, column)
        
    def ConMap2Mask_prob(self,c_map):
        c_map = c_map.view(c_map.shape[0],self.class_num,8,c_map.shape[2],c_map.shape[3])
        batch,class_num,channel, row, column = c_map.size()


        shifted_c_map = torch.zeros(c_map.size()).cuda()

        for i in range(8):
            shifted_c_map[:,:,i] = self.shift_diag(c_map[:,:,7-i].clone(),Directable[TL_table[i]])
        vote_out = c_map*shifted_c_map
        vote_out = vote_out.float()
        pred_mask,_ = torch.max(vote_out,dim=2)

        return pred_mask,vote_out
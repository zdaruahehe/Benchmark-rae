# -*- coding: utf-8 -*-
"""
Created on August 10 18:54:50 2023

@author: Yuhan

WAE Losses functions and classes

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from network.wae.wae_mmd_config import WAE_MMD_Config

class KernelLoss(nn.Module):
    def forward(self, X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int,
               kernel = 'imq'):
        
        batch_size = X.size(0)
    
        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
    
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
    
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
        stats = 0
        
        if kernel == 'imq':
            
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                # Here this is already hardcoded for sigma = 1, assuming normal
                C = 2 * h_dim * 1.0 * scale
                res1 = C / (C + dists_x)
                res1 += C / (C + dists_y)
        
                if torch.cuda.is_available():
                    res1 = (1 - torch.eye(batch_size).cuda()) * res1
                else:
                    res1 = (1 - torch.eye(batch_size)) * res1
        
                res1 = res1.sum() / (batch_size - 1)
                res2 = C / (C + dists_c)
                res2 = res2.sum() * 2. / (batch_size)
                stats += res1 - res2
        else:
            
            for scale in [.1, .2, .5, 1., 2., 5., 10.]:
                C = 2 * h_dim * 1.0 / scale
                res1 = torch.exp(-C * dists_x)
                res1 += torch.exp(-C * dists_y)
    
                if torch.cuda.is_available():
                    res1 = (1 - torch.eye(batch_size).cuda()) * res1
                else:
                    res1 = (1 - torch.eye(batch_size)) * res1
    
            res1 = res1.sum() / (batch_size - 1)
            res2 = torch.exp(-C * dists_c)
            res2 = res2.sum() * 2. / batch_size
            stats += res1 - res2
        return stats

class ImqKernelLoss(nn.Module):
    def forward(self, X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
        
        batch_size = X.size(0)
    
        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
    
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
    
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
        stats = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            # Here this is already hardcoded for sigma = 1, assuming normal
            C = 2 * h_dim * 1.0 * scale
            res1 = C / (C + dists_x)
            res1 += C / (C + dists_y)
    
            if torch.cuda.is_available():
                res1 = (1 - torch.eye(batch_size).cuda()) * res1
            else:
                res1 = (1 - torch.eye(batch_size)) * res1
    
            res1 = res1.sum() / (batch_size - 1)
            res2 = C / (C + dists_c)
            res2 = res2.sum() * 2. / (batch_size)
            stats += res1 - res2
    
        return stats

class RbfKenelLoss(nn.Module):
    def forward(self, X: torch.Tensor,
                Y: torch.Tensor,
                h_dim: int):
    
        batch_size = X.size(0)

        norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_x = torch.mm(X, X.t())  # batch_size x batch_size
        dists_x = norms_x + norms_x.t() - 2 * prods_x
    
        norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
        prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
        dists_y = norms_y + norms_y.t() - 2 * prods_y
    
        dot_prd = torch.mm(X, Y.t())
        dists_c = norms_x + norms_y.t() - 2 * dot_prd
    
        stats = 0
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = 2 * h_dim * 1.0 / scale
            res1 = torch.exp(-C * dists_x)
            res1 += torch.exp(-C * dists_y)
    
            if torch.cuda.is_available():
                res1 = (1 - torch.eye(batch_size).cuda()) * res1
            else:
                res1 = (1 - torch.eye(batch_size)) * res1
    
            res1 = res1.sum() / (batch_size - 1)
            res2 = torch.exp(-C * dists_c)
            res2 = res2.sum() * 2. / batch_size
            stats += res1 - res2
    
        return stats
    
def get_loss_from_name_WAE(name):
    if name == "l1":
        return L1LossWrapper()
    elif name == 'l2':
        return L2LossWrapper()
    else:
        return per_pix_recon_loss()

class L1LossWrapper(nn.Module):
    '''
    Following the loss function in https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    '''
    def __init__(self):
        super(L1LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        l1loss = torch.sum(torch.abs(pred_img - gt_img), dim=[1,2,3])
        l1loss = 0.02 * l1loss.mean()
        return l1loss
    
class L2LossWrapper(nn.Module):
    def __init__(self):
        super(L2LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean((pred_img - gt_img) ** 2, dim=1)  
    

class per_pix_recon_loss(nn.Module):
    '''
    Following the loss function in https://github.com/tolstikhin/wae/blob/63515656201eb6e3c3f32f6d38267401ed8ade8f/wae.py
    '''
    def __init__(self):
        super(per_pix_recon_loss, self).__init__()

    def forward(self, pred_img, gt_img):
        loss = torch.sum((pred_img - gt_img) ** 2, dim=[1,2,3])
        loss = 0.05 * loss.mean()
        return loss

    

class WAETotalLoss(nn.Module):
    '''
    Depends on get_loss_from_name function and associated Loss wrapper classes above
    Only IMQ Kernel implemented
    '''
    def __init__(self,device,
                 mmd_weight=1, recon_loss_name = 'l2'):
        super(WAETotalLoss, self).__init__()
        
        # Initialise recon loss and the other losses
        self.recon_loss = get_loss_from_name_WAE(recon_loss_name)
        self.KernelLoss = KernelLoss()
        """
        if kernel_choice == 'imq':
            self.kernel = ImqKernelLoss()
        else:
            self.kernel = RbfKenelLoss()
        """
        self.mmd_weight = mmd_weight
        
        self.device = device
        
    
    def forward(self, pred_img, gt_img, embedding, latent_dims, sigma = 1, kernel_choice = 'imq'):
        ''' 
        gt_img: original input image
        embedding: z
        latent_dims: latent dimensions of z
        sigma: std deviation
        '''
        
        z_fake = Variable(torch.randn(gt_img.size()[0], latent_dims) * sigma).to(self.device)
        z_real = embedding.to(self.device)
        
        mmd_loss = self.KernelLoss(z_real, z_fake, h_dim=latent_dims, kernel = kernel_choice)
        mmd_loss = mmd_loss # / batch_size, to see if need to divide by batch size
        
        
        recon_loss = self.recon_loss(pred_img, gt_img).mean()
        
        total_loss = recon_loss + mmd_loss*self.mmd_weight
        
        return total_loss
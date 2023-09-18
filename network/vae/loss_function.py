import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_loss_from_name(name):
    if name == "l1":
        return L1LossWrapper()
    elif name == 'l2':
        return L2LossWrapper()
    else:
        return L2LossWrapper()
    
class L1LossWrapper(nn.Module):
    def __init__(self):
        super(L1LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean(torch.abs(pred_img - gt_img), dim=1)


class L2LossWrapper(nn.Module):
    def __init__(self):
        super(L2LossWrapper, self).__init__()

    def forward(self, pred_img, gt_img):
        return torch.mean((pred_img - gt_img) ** 2, dim=1)

class LossKLDiverg(nn.Module):
    def forward(self, mu, log_sigma):
        # print(mu.shape)
        return 0.5*torch.sum(torch.exp(log_sigma*2) + mu.pow(2) -2*log_sigma  - 1, dim = 1)
    
    
class VAETotalLoss(nn.Module):
    '''
    Depends on associated Loss wrapper classes above
    '''
    def __init__(self,
                 kl_weight=0.312, recon_loss_func=None):
        super(VAETotalLoss, self).__init__()

        self.kl_weight = kl_weight
        self.kl_diverg = LossKLDiverg()
        
        # Initialise kl_loss and recon_loss
        # self.kl_loss = 0
        # self.recon_loss = 0
        
        # By default uses the mse
        if recon_loss_func is None:
            self.lossfunc = L2LossWrapper()
        else:
            self.lossfunc = recon_loss_func
        
    def forward(self, pred_img, gt_img, mu, log_sigma):

        # print('prediction size', pred_img.shape)
        
        # This is by default the MSE implementation
        self.recon_loss = self.lossfunc(pred_img, gt_img).sum(dim=[1, 2])
        
        # Compute KL divergence loss
        self.kl_loss = self.kl_diverg(mu, log_sigma)
        
        #print('mean recon_loss: ', self.recon_loss.mean())
        #print('mean kl_loss: ', self.kl_loss.mean())
        
        # Add the KL divergence loss to reconstruction loss
        loss = self.recon_loss + self.kl_loss * self.kl_weight
        
        return loss.mean()
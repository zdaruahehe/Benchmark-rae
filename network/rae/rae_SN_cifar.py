# -*- coding: utf-8 -*-

import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.nn import ModuleList

import os

from skimage.io import imread
#from opts.opts import TrainOptions, INFO

import copy
from tqdm import tqdm

from torchvision.utils import save_image
from ..SN_layers.SNConv2d import *

# =========================================================================
#   Define Encoder
# =========================================================================

class SN_Encoder(nn.Module):
    def __init__(self, latent_dims, SN = True):
        super(SN_Encoder, self).__init__()      
        
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # nn.LeakyReLU(0.1)
        )
        self.linear = nn.Linear(4096, latent_dims) #  128*72 = 9216  
     
        self.sn_conv_layers = nn.Sequential(
            
            SNConv2d(3, out_channels = 128, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(128, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(256, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConv2d(512, out_channels = 1024, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(1024),
            nn.ReLU()
            # nn.LeakyReLU(0.02)
        )
        

        self.sn_linear = SNLinear(4096, latent_dims)
        
        self.SN = SN

    def forward(self, x):
        
        
        # Check if spectral norm is implemented
        if self.SN:
            x = self.sn_conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.sn_linear(x)
        else:
            x = self.conv_layers(x)
            x = torch.flatten(x, start_dim=1)
            z = self.linear(x)
        
        """       
        x = self.sn_conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        z = self.sn_linear(x)    
        """   
        
        return z
    
# =========================================================================
#   Define Decoder
# =========================================================================

class SN_Decoder(nn.Module):
    def __init__(self, latent_dims, SN = True):
        super(SN_Decoder, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dims, 8*8*1024)
        )
        self.batch_norm1 = nn.BatchNorm2d(1024)
        
        
        self.conv_layers = nn.Sequential(
            
            nn.ConvTranspose2d(1024, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(512, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, out_channels = 3, kernel_size = (5,5), stride = (1,1), padding = 2)
        )
        
        
        self.sn_conv_layers = nn.Sequential(
            
            SNConvTranspose2d(1024, out_channels = 512, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            SNConvTranspose2d(512, out_channels = 256, kernel_size = (4,4), stride = (2,2), padding = (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.LeakyReLU(0.02),
            
            # Using kernel size of 9 by 9 instead to get 32
            SNConvTranspose2d(256, out_channels = 3, kernel_size = (5,5), stride = (1,1), padding = 2)
        )

        
        self.sn_fc_layer = nn.Sequential(
            SNLinear(latent_dims, 8*8*1024)
        )
        
        self.SN = SN

    def forward(self, z):

        # Check if spectral norm is implemented
        if self.SN:
            x = self.sn_fc_layer(z)
            x = x.view(x.size(0),1024,8,8)
            x = F.relu(self.batch_norm1(x))
            x = self.sn_conv_layers(x)

        else:
            x = self.fc_layer(z)
            x = x.view(x.size(0),1024,8,8)
            # print(x.shape)
            x = F.relu(self.batch_norm1(x))
            x = self.conv_layers(x)
        
        x = F.sigmoid(x)
        # print(x.shape)
        return x
    
class SN_RAE(nn.Module):
    def __init__(self, latent_dims=32, SN = True):
        super(SN_RAE, self).__init__()
        self.encoder = SN_Encoder(latent_dims, SN)
        self.decoder = SN_Decoder(latent_dims, SN)
            

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)
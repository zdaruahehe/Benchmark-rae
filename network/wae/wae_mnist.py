import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from network.wae.wae_mmd_config import WAE_MMD_Config

from typing import Optional

class Encoder(nn.Module):
    def __init__(self,
                 num_filters=128,
                 bottleneck_size=16,
                 include_batch_norm=True):

        super(Encoder, self).__init__()

        self.include_bn = include_batch_norm

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_filters,
                               kernel_size=4,
                               stride=2,
                               padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(in_channels=num_filters,
                               out_channels=num_filters * 2,
                               kernel_size=4,
                               stride=2,
                               padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_filters * 2)

        self.conv3 = nn.Conv2d(in_channels=num_filters * 2,
                               out_channels=num_filters * 4,
                               kernel_size=4,
                               stride=2,
                               padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(num_filters * 4)

        self.conv4 = nn.Conv2d(in_channels=num_filters * 4,
                               out_channels=num_filters * 8,
                               kernel_size=4,
                               stride=2,
                               padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(num_filters * 8)

        self.fc = nn.Linear(4096, bottleneck_size)

    def forward(self, x):

        x = self.conv1(x)
        if self.include_bn:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.include_bn:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        if self.include_bn:
            x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        if self.include_bn:
            x = self.bn4(x)
        x = F.relu(x)

        x = torch.flatten(x, start_dim=1)
        z = self.fc(x)

        return z


# =========================================================================
#   Define Decoder
# =========================================================================

class Decoder(nn.Module):
    def __init__(self,
                 num_filters=128,
                 bottleneck_size=16,
                 include_batch_norm=True):

        super(Decoder, self).__init__()

        self.include_bn = include_batch_norm
        
        self.fc = nn.Sequential(
            nn.Linear(bottleneck_size, 8*8*1024)
        )
        
        self.batch_norm1 = nn.BatchNorm2d(1024)

        self.conv1 = nn.ConvTranspose2d(in_channels=1024,
                                        out_channels=num_filters * 4,
                                        kernel_size=4,
                                        stride=2,
                                        padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(num_filters * 4)

        self.conv2 = nn.ConvTranspose2d(in_channels=num_filters * 4,
                                        out_channels=num_filters * 2,
                                        kernel_size=4,
                                        stride=2,
                                        padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_filters * 2)

        self.conv3 = nn.ConvTranspose2d(in_channels=num_filters * 2,
                                        out_channels=1,
                                        kernel_size=5,
                                        stride=1,
                                        padding=(2, 2))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 8, 8)
        x = F.relu(self.batch_norm1(x))
        
        x = self.conv1(x)
        if self.include_bn:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        if self.include_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)

        return x


class WAE(nn.Module):

    def __init__(self, latent_dims=16):
        super(WAE, self).__init__()

        self.encoder = Encoder(bottleneck_size=latent_dims)
        self.decoder = Decoder(bottleneck_size=latent_dims)
        
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        self.z = self.encoder(x)
        recon_x = self.decoder(self.z)
        return recon_x

"""
Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


def loss_fn(x, x_reconst, mu, logvar, beta=0.024):
    # reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconst_loss + (beta * kl_loss)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class Encoder(nn.Module):
    def __init__(self, num_filters=128,
                 bottleneck_size=128,
                 include_batch_norm=True,
                 constant_sigma = None):
        super(Encoder, self).__init__()

        self.include_bn = include_batch_norm
        self.constant_sigma = constant_sigma

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=num_filters, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters * 2, out_channels=num_filters * 4, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_filters * 4, out_channels=num_filters * 8, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 8),
            nn.ReLU(),
        )

        self.mu_layer = nn.Linear(4096, bottleneck_size)
        
        self.dense_for_sigma = nn.Sequential(     
            nn.Linear(4096, bottleneck_size),
            nn.Tanh()
            )

    def forward(self, x):
        # Convnet.
        x = self.conv_net(x)

        # Flatten.
        x = torch.flatten(x, start_dim=1)

        # Fully connected.
        mu = self.mu_layer(x)
        if self.constant_sigma is None:
            log_sigma = 5*self.dense_for_sigma(x)
            
        else:
            # constant variance variant, using similar method as in the keras implementation
            log_sigma = LambdaLayer(lambda var: torch.log(self.constant_sigma))(x)
            
        #logvar = self.logvar_layer(x)

        return mu, log_sigma


class Decoder(nn.Module):
    def __init__(self, num_filters=128,
                 bottleneck_size=128,
                 include_batch_norm=True):
        super(Decoder, self).__init__()

        self.include_bn = include_batch_norm
        
        #self.latent_dim = latent_dim
        #self.hidden_dim = hidden_dim
        self.fc_layer = nn.Sequential(
            nn.Linear(bottleneck_size, 8*8*1024)
        )
        
        self.batch_norm1 = nn.BatchNorm2d(1024)

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=num_filters * 4, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=num_filters * 4, out_channels=num_filters * 2, kernel_size=4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
        )
        
        self.conv = nn.ConvTranspose2d(in_channels=num_filters * 2,
                                        out_channels=3,
                                        kernel_size=5,
                                        stride=1,
                                        padding=(2, 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # FC from latent to hidden dim.
        x = self.fc_layer(z)

        # Unflatten.
        x = x.view(x.size(0), 1024, 8, 8)
        x = F.relu(self.batch_norm1(x))

        # Convnet.
        x = self.deconv_net(x)
        x = self.conv(x)
        x = self.sigmoid(x)

        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
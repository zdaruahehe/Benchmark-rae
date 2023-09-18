# -*- coding: utf-8 -*-

"""
    @date:   2022.7.24 
    @author: Yuhan
    @target: WAE trainer
"""
import os
import time
import itertools

import torch
import torch.optim as optim
import matplotlib.pyplot as plt


from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np


# import util

from opts.opts import TrainOptions
from utils.libs import LatentSpaceSampler
from utils.utils import SaveModel
import network.vae.vae_mnist as vae
#from network.vae.vae_mnist import VAE
from network.vae.vae_cifar import VAE
from network.vae.loss_function import *

from utils import save_batches_of_images


def main(opts):
    # Set variables.
    kl_weight = 0.024
    learning_rate = 0.001
    

    # fix seed for experiment.
    # util.fix_seed()

    # Load Dataset.
    """   
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    
    train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)
    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))

    data_test = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=opts.batch_size,
                                                    shuffle=True)
    
    data_loader_valadation = torch.utils.data.DataLoader(dataset=data_validation,
                                                    batch_size=opts.batch_size,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True)
    """
    """
    # transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32)])
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
    
    train = datasets.FashionMNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)
    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))

    data_test = datasets.FashionMNIST(root="./data/",
                               transform=transform,
                               train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=opts.batch_size,
                                                    shuffle=True)
    
    data_loader_valadation = torch.utils.data.DataLoader(dataset=data_validation,
                                                    batch_size=opts.batch_size,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True)
    
    """
    
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32)])
    
    train = datasets.CIFAR10(root="./data/", transform=transform, train=True, download=True)
    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))

    data_test = datasets.CIFAR10(root="./data/", transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=opts.batch_size, shuffle=True)

    
    data_loader_valadation = torch.utils.data.DataLoader(dataset=data_validation, batch_size=opts.batch_size, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=opts.batch_size, shuffle=True)
    

    # Load Encoder, Decoder.
    vae_net = VAE()
    vae_net.to(opts.device)

    # Set loss fn.
    loss_fn = VAETotalLoss(kl_weight)
    
    # Load optimizer.
    
    optimizer = optim.Adam(vae_net.parameters(), lr=learning_rate) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    train_loss_list = []
    val_loss_list = []
    
    # Epoch loop
    for epoch in range(0, opts.epoch):
        # print('Epoch {} of {}'.format(epoch, num_epochs))
        
        train_bar = data_loader_train
        test_bar = data_loader_test
        valadation_bar = data_loader_valadation
        
        vae_net.train()
        
        running_loss = 0.0

        # Batch loop.
        for i, data in enumerate(train_bar):
            # print('Batch {}'.format(i_batch+1))

            # Load batch.
            x, y = data
            x = x.to(opts.device)

            # Reset gradient.
            optimizer.zero_grad()          

            # Run batch, calculate loss, and backprop.
            x_reconst, mu, log_sigma = vae_net(x)
            loss1 = loss_fn(x_reconst, x, mu, log_sigma)
            loss1.to(opts.device)
            running_loss += loss1.item()
            loss1.backward()
            optimizer.step()
        train_loss = running_loss / len(train_bar.dataset)       
        train_loss_list.append(train_loss)    
            
            #train_bar.set_description(
                #"Epoch {} [{}, {}]".format(epoch, i + 1, len(data_loader_train)))

        running_loss = 0.0
        
        vae_net.eval()
        
        with torch.no_grad():
            for j, val_data in enumerate(valadation_bar):
                    X_val, y_val = val_data
                    X_val = X_val.to(opts.device)
                    x_hat, mu, log_sigma = vae_net(X_val.float())
                    loss2 = loss_fn(x_hat, X_val, mu, log_sigma)
                    loss2.to(opts.device)
                    running_loss += loss2.item()
        val_loss = running_loss / len(valadation_bar.dataset)
        val_loss_list.append(val_loss)
        
        print(f"Epoch {epoch+1} of {opts.epoch}")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Val Loss: {val_loss:.8f}")
        print()
            
        if epoch == 0 or epoch == 99:
            """
            with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'images'), outputs)
            
              
            for i, data in enumerate(valadation_bar):
                encoder.eval()
                decoder.eval()

                X_test, y_test = data
                X_test = X_test.to(opts.device)
                outputs = decoder(encoder(X_test))
            """
                    
            images = None
            original_img = None
            sampled_images = None
            while True:
                
                for i, data in enumerate(valadation_bar):
                    
                    X_test, y_test = data
                    X_test = X_test.to(opts.device)
                    outputs = vae_net(X_test)
                    # self.original_img[i*batch_size:(i+1)*batch_size, :] = backup_test_gen.next()[0][0]
                    temp_original = X_test
                    
                    if original_img is None:
                        original_img = temp_original.cpu().numpy()
                        img, _, _ = vae_net(X_test)   
                        images = img.cpu().detach().numpy()                                              
                        latentspace = LatentSpaceSampler(vae_net.encoder)
                        zs = latentspace.get_zs(temp_original)
                        zs = torch.from_numpy(zs).float().to(opts.device)
                        sampled_images = vae_net.decode(zs).cpu().detach().numpy()
                    else:
                        original_img = np.concatenate((original_img, temp_original.cpu().numpy()), axis=0)
                        img, _, _ = vae_net(X_test)
                        images = np.concatenate((images, img.cpu().detach().numpy()), axis=0)
                        if sampled_images is not None:
                            latentspace = LatentSpaceSampler(vae_net.encoder)
                            zs = latentspace.get_zs(temp_original)
                            zs = torch.from_numpy(zs).float().to(opts.device)
                            temp_sampled = vae_net.decode(zs)
                            sampled_images = np.concatenate((sampled_images, temp_sampled.cpu().detach().numpy()),
                                                                axis=0)
                    if original_img.shape[0] >= 10000:
                        break
                if original_img.shape[0] >= 10000:
                    break
                   
                    
            original_img = original_img[:10000]
            images = images[:10000]
            sampled_images = sampled_images[:10000]
            images = torch.from_numpy(images).float().to(opts.device)
            sampled_images = torch.from_numpy(sampled_images).float().to(opts.device)
            with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'images', 'reconstructed'), images)
            with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'images', 'samples'), sampled_images)
            
            torch.save({'model_state_dict': vae_net.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join('./', '_params_{}.pt'.format(epoch)))
            
        scheduler.step(val_loss)
        
        # util.save_weights(vae_net, os.path.join(save_dir, 'vae_{}.pth'.format(epoch)))


if __name__ == '__main__':
    opts = TrainOptions().parse()
    main(opts)
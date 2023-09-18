# -*- coding: utf-8 -*-

"""
    @date:   2022.7.30 
    @author: Yuhan
    @target: Training script for vae, beta-vae.
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
from network.wae.wae_mnist import WAE
#from network.wae.wae_cifar import WAE
from network.wae.loss_function import *
from utils import save_batches_of_images


def main(opts):
    # Set variables.
    kernel = 'imq'
    latent_dim = 16
    num_epochs = 100
    beta = 0.312
    cuda = True
    learning_rate = 0.001
    adaptive = True  # True
    embedding_weight = 3e-2

    # fix seed for experiment.
    # util.fix_seed()

    # Load Dataset.
    
    
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
    
    
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32)])
    
    train = datasets.CIFAR10(root="./data/", transform=transform, train=True, download=True)
    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))

    data_test = datasets.CIFAR10(root="./data/", transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=opts.batch_size, shuffle=True)

    
    data_loader_valadation = torch.utils.data.DataLoader(dataset=data_validation, batch_size=opts.batch_size, shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=opts.batch_size, shuffle=True)

    """
    
    # Load Model
    wae_net = WAE(latent_dims=latent_dim)
    wae_net.to(opts.device)
    
    # Set loss fn.
    loss_fn = WAETotalLoss(device=opts.device, mmd_weight=embedding_weight, recon_loss_name='l2')

    # Load optimizer.
    
    optimizer = optim.Adam(wae_net.parameters(), lr=learning_rate)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    
    train_loss_list = []
    val_loss_list = []

    # Epoch loop
    for epoch in range(0, opts.epoch):
        # print('Epoch {} of {}'.format(epoch, num_epochs))
        # start = time.time()
        
        train_bar = data_loader_train
        test_bar = data_loader_test
        valadation_bar = data_loader_valadation
        
        wae_net.train()
        
        running_loss = 0.0

        # Batch loop.
        for i, data in enumerate(train_bar):

            # Load batch.
            x, y = data
            x = x.to(opts.device)

            # Reset gradient.
            optimizer.zero_grad()           

            # Run batch, calculate loss, and backprop.
            z = wae_net.encoder(x.float())
            x_hat = wae_net.decoder(z.float())
            
            loss1 = loss_fn(x_hat, x, z, latent_dim,  kernel_choice = kernel)
            loss1.to(opts.device)
            running_loss += loss1.item()
            loss1.backward()
            optimizer.step()
        train_loss = running_loss / len(train_bar.dataset)       
        train_loss_list.append(train_loss)
            
        running_loss = 0.0
        
        wae_net.eval()
        
        with torch.no_grad():
            for j, val_data in enumerate(valadation_bar):
                    X_val, y_val = val_data
                    X_val = X_val.to(opts.device)
                    z2 = wae_net.encoder(X_val.float())
                    x_recon = wae_net.decoder(z2.float())
                    loss2 = loss_fn(x_recon, X_val, z2, latent_dim, kernel_choice = kernel)
                    loss2.to(opts.device)
                    running_loss += loss2.item()
        val_loss = running_loss / len(valadation_bar.dataset)
        val_loss_list.append(val_loss)
        
        print(f"Epoch {epoch+1} of {opts.epoch}")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Val Loss: {val_loss:.8f}")
        print()
            
        if epoch == 10 or epoch == (opts.epoch - 1):
                    
            images = None
            original_img = None
            sampled_images = None
            #model_first_parameters = next(wae_net.decoder.parameters())
            #model_input_shape = model_first_parameters.shape
                
            while True:    
                for i, data in enumerate(train_bar):
                    X_test, y_test = data
                    X_test = X_test.to(opts.device)
                    outputs = wae_net(X_test)
                    # self.original_img[i*batch_size:(i+1)*batch_size, :] = backup_test_gen.next()[0][0]
                    temp_original = X_test 

                    if original_img is None:
                        original_img = temp_original.cpu().numpy()
                        img = wae_net(X_test)   
                        images = img.cpu().detach().numpy()
                        
                        dim_z = wae_net.encoder(temp_original).size(-1)
                        z_origina_shape = (temp_original.shape[0], dim_z)
                        num_smpls = temp_original.shape[0]
                        
                        try:
                            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_origina_shape[1:]), ), cov=np.eye(dim_z),
                                                         size=num_smpls)  # https://www.zhihu.com/question/288946037/answer/649328934
                        except np.linalg.LinAlgError as e:
                            print(np.eye(dim_z))
                            print(e)
                            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_origina_shape[1:]), ),
                                                         cov=np.eye(dim_z) + 1e-5 * np.eye(np.eye(dim_z).shape[0]),
                                                         size=num_smpls)
                        zs = np.reshape(zs_flattened, (num_smpls,) + z_origina_shape[1:])
                        """                                            
                        randnorm_tensor = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(temp_original.shape[0], list(model_input_shape)[1]))).to(torch.float32).to(opts.device)
                        sampled_images = wae_net.decoder(randnorm_tensor).cpu().detach().numpy()
                        """                                               
                        #latentspace = LatentSpaceSampler(wae_net.encoder)
                        #zs = latentspace.get_zs(temp_original)
                        zs = torch.from_numpy(zs).float().to(opts.device)
                        sampled_images = wae_net.decode(zs).cpu().detach().numpy()
                        
                    else:
                        original_img = np.concatenate((original_img, temp_original.cpu().numpy()), axis=0)
                        img= wae_net(X_test)
                        images = np.concatenate((images, img.cpu().detach().numpy()), axis=0)
                        if sampled_images is not None:
                            """
                            randnorm_tensor = torch.from_numpy(np.random.normal(loc=0.0, scale=1.0, size=(temp_original.shape[0], list(model_input_shape)[1]))).to(torch.float32).to(opts.device)    
                            temp_sampled = wae_net.decode(randnorm_tensor)
                            """
                            #latentspace = LatentSpaceSampler(wae_net.encoder)
                            #zs = latentspace.get_zs(temp_original)
                            dim_z = wae_net.encoder(temp_original).size(-1)
                            z_origina_shape = (temp_original.shape[0], dim_z)
                            num_smpls = temp_original.shape[0]
                            try:
                                zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_origina_shape[1:]), ), cov=np.eye(dim_z),
                                                         size=num_smpls)  # https://www.zhihu.com/question/288946037/answer/649328934
                            except np.linalg.LinAlgError as e:
                                print(np.eye(dim_z))
                                print(e)
                                zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_origina_shape[1:]), ),
                                                         cov=np.eye(dim_z) + 1e-5 * np.eye(np.eye(dim_z).shape[0]),
                                                         size=num_smpls)
                            zs = np.reshape(zs_flattened, (num_smpls,) + z_origina_shape[1:])
                            
                            zs = torch.from_numpy(zs).float().to(opts.device)                           
                            temp_sampled = wae_net.decode(zs)
                            sampled_images = np.concatenate((sampled_images, temp_sampled.cpu().detach().numpy()),
                                                                axis=0)
                    if original_img.shape[0] >= 10000:
                        break
                if original_img.shape[0] >= 10000:
                    break
                   
                    
            original_img = original_img[:10000]
            images = images[:10000]
            # print(images[0])
            sampled_images = sampled_images[:10000]
            images = torch.from_numpy(images).float().to(opts.device)
            sampled_images = torch.from_numpy(sampled_images).float().to(opts.device)
            with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'images', 'reconstructed'), images)
            with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'images', 'samples'), sampled_images)
            torch.save({'model_state_dict': wae_net.state_dict(),
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
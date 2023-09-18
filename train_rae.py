# -*- coding: utf-8 -*-

"""
    @date:   2022.7.22 
    @author: Yuhan
    @target: RAE trainer
"""
import itertools
import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from network.rae.rae_mnist import Encoder, Decoder
#from network.rae.rae_cifar import Encoder, Decoder
#from network.rae.rae_SN_mnist import SN_Encoder, SN_Decoder
#from network.rae.rae_SN_cifar import SN_Encoder, SN_Decoder

from opts.opts import TrainOptions
from utils.libs import LatentSpaceSampler
from network.rae.loss_function import *
from utils.utils import SaveModel
from utils import save_batches_of_images

def get_loss_reg(model):
    return sum(parameter.square().sum() for parameter in model.parameters())

def main(opts):
    # 0) 创建数据集
    # Create the data loader
    
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
    """
    
    # Create the model
    #encoder = SN_Encoder(latent_dims = 128)
    #decoder = SN_Decoder(latent_dims = 128)
    encoder = Encoder()
    decoder = Decoder()
    encoder.to(opts.device)
    decoder.to(opts.device)

    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(),
                                                 decoder.parameters()),
                                 lr=0.001)
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)    
    
    loss_calc = TotalLoss(loss=opts.loss_choice)
    #loss_calc = TotalLoss(regularization_loss=L2RegLoss(), loss=opts.loss_choice)
    
    regularisation_loss_type = 'AE'
    
    train_loss_list = []
    val_loss_list = []
    
    for ep in range(0, opts.epoch):
        
        #train_bar = tqdm(data_loader_train)
        #test_bar = tqdm(data_loader_test)
        #valadation_bar = tqdm(data_loader_valadation)
        
        train_bar = data_loader_train
        test_bar = data_loader_test
        valadation_bar = data_loader_valadation
        
        
        #######################################
        # Train                
        #######################################
        
        # scheduler.step(ep)

        encoder.train()
        decoder.train()
        
        running_loss = 0.0

        for i, data in enumerate(train_bar):
            
            X_train, y_train = data
            X_train = X_train.to(opts.device)
            
            optimizer.zero_grad()
            
            embeddings = encoder(X_train.float())
            output = decoder(embeddings)

            loss1 = loss_calc(output, X_train, embeddings, decoder.parameters())
            #loss_stat_list[i] = loss1
            loss1.to(opts.device)
            running_loss += loss1.item()           
            loss1.backward()
            optimizer.step()

            # Output training stats
            #train_bar.set_description(
                #"Epoch {} [{}, {}] [Total Loss] {}".format(ep, i + 1, len(data_loader_train), loss_stat_list[-1]))
            #train_bar.set_description(
                #"Epoch {} [{}, {}]".format(ep, i + 1, len(data_loader_train)))  
        train_loss = running_loss / len(train_bar.dataset)       
        train_loss_list.append(train_loss)
        
        #######################################
        # Valid                
        #######################################
        
        running_loss = 0.0    
            
        encoder.eval()
        decoder.eval()
        
        if regularisation_loss_type != 'grad_pen':
            with torch.no_grad():
                for j, val_data in enumerate(valadation_bar):
                    X_val, y_val = val_data
                    X_val = X_val.to(opts.device)
                    out = decoder(encoder(X_val.float()))
                    loss2 = loss_calc(out, X_val, encoder(X_val.float()), decoder.parameters())
                    loss2.to(opts.device)
                    running_loss += loss2.item()
        else:
            for j, val_data in enumerate(valadation_bar):
                X_val, y_val = val_data
                X_val = X_val.to(opts.device)
                z = encoder(X_val.float())
                out = decoder(z.float())
                #out = decoder(encoder(X_val.float()).float())
                loss2 = loss_calc(out, X_val, z, decoder.parameters())
                loss2.to(opts.device)
                running_loss += loss2.item()
                
        
        val_loss = running_loss / len(valadation_bar.dataset)
        val_loss_list.append(val_loss)
        print(f"Epoch {ep+1} of {opts.epoch}")
        print(f"Train Loss: {train_loss:.8f}")
        print(f"Val Loss: {val_loss:.8f}")
        print()
                
        if ep == (opts.epoch - opts.epoch) or ep == (opts.epoch - 1):
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
                    outputs = decoder(encoder(X_test))
                    # self.original_img[i*batch_size:(i+1)*batch_size, :] = backup_test_gen.next()[0][0]
                    temp_original = X_test
                    if original_img is None:
                        original_img = temp_original.cpu().numpy()
                        images = decoder(encoder(X_test)).cpu().detach().numpy()
                        latentspace = LatentSpaceSampler(encoder)
                        zs = latentspace.get_zs(temp_original)
                        zs = torch.from_numpy(zs).float().to(opts.device)
                        sampled_images = decoder(zs).cpu().detach().numpy()
                    else:
                        original_img = np.concatenate((original_img, temp_original.cpu().numpy()), axis=0)
                        images = np.concatenate((images, decoder(encoder(X_test)).cpu().detach().numpy()), axis=0)
                        if sampled_images is not None:
                            latentspace = LatentSpaceSampler(encoder)
                            zs = latentspace.get_zs(temp_original)
                            zs = torch.from_numpy(zs).float().to(opts.device)
                            temp_sampled = decoder(zs)
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
                
        
        scheduler.step(val_loss)
        
        if ep == (opts.epoch - 1):    
            SaveModel(encoder, decoder, train_loss_list, val_loss_list, optimizer, scheduler, dir='./', epoch=ep)


if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)
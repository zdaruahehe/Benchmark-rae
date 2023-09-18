import torch
import torch.optim as optim
import matplotlib.pyplot as plt


from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
from sklearn import mixture

import logging

logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

from network.rae.rae_mnist import Encoder, Decoder
#from network.rae.rae_cifar import Encoder, Decoder
from network.vae.vae_mnist import VAE
#from network.vae.vae_cifar import VAE
from network.wae.wae_mnist import WAE
#from network.wae.wae_cifar import WAE
from opts.opts import TrainOptions
from utils import save_batches_of_images

class GaussianMixtureSampler():
    
    def __init__(self, model=None, encoder=None, decoder=None, n_components=None):
        self.model = model
        self.encoder = encoder
        self.decoder = decoder
        self.n_components = n_components

    def fit(self, data_train, opts, isVAE = False):
        """Method to fit the sampler from the training data
        Args:
            data_train (torch.Tensor): The train data needed to retreive the training embeddings
                    and fit the mixture in the latent space. Must be of shape n_imgs x im_channels x ...
                    and in range [0-1]
        """
        self.is_fitted = True       
        
        data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=False)

        mu = []

        with torch.no_grad():
            train_bar = tqdm(data_loader_train)
            for _, inputs in enumerate(train_bar):
                """
                mu_data = self.model.encoder(inputs["data"].to(self.device))[
                    "embedding"
                ]
                """
                x,_ = inputs
                x = x.to(opts.device)
                if isVAE == True:
                    z = self.model.encode(x)
                else:
                    z = self.encoder(x)
                
                mu.append(z)

        mu = torch.cat(mu)

        if self.n_components > mu.shape[0]:
            self.n_components = mu.shape[0]
            logger.warning(f"Setting the number of component to {mu.shape[0]} since"
                 "n_components > n_samples when fitting the gmm")

        gmm = mixture.GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=500,
            verbose=2,
            tol=1e-3,
        )
        gmm.fit(mu.cpu().detach())

        self.gmm = gmm

    def sample(
        self,
        num_samples: int = 1,
        batch_size: int = 200,
        output_dir: str = None,
        return_gen: bool = True,
        save_sampler_config: bool = False,
    ) -> torch.Tensor:
        """Main sampling function of the sampler.
        Args:
            num_samples (int): The number of samples to generate
            batch_size (int): The batch size to use during sampling
            output_dir (str): The directory where the images will be saved. If does not exist the 
                folder is created. If None: the images are not saved. Defaults: None.
            return_gen (bool): Whether the sampler should directly return a tensor of generated 
                data. Default: True.
            save_sampler_config (bool): Whether to save the sampler config. It is saved in 
                output_dir
        
        Returns:
            ~torch.Tensor: The generated images
        """

        if not self.is_fitted:
            raise ArithmeticError(
                "The sampler needs to be fitted by calling smapler.fit() method"
                "before sampling."
            )

        full_batch_nbr = int(num_samples / batch_size)
        last_batch_samples_nbr = num_samples % batch_size

        x_gen_list = []

        for i in range(full_batch_nbr):

            z = (
                torch.tensor(self.gmm.sample(batch_size)[0])
                .to(opts.device)
                .type(torch.float)
            )
            # x_gen = self.model.decoder(z)["reconstruction"].detach()
            x_gen = self.decoder(z).detach()

            if output_dir is not None:
                for j in range(batch_size):
                    self.save_img(
                        x_gen[j], output_dir, "%08d.png" % int(batch_size * i + j)
                    )

            x_gen_list.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = (
                torch.tensor(self.gmm.sample(last_batch_samples_nbr)[0])
                .to(opts.device)
                .type(torch.float)
            )
            x_gen = self.decoder(z).detach()

            if output_dir is not None:
                for j in range(last_batch_samples_nbr):
                    self.save_img(
                        x_gen[j],
                        output_dir,
                        "%08d.png" % int(batch_size * full_batch_nbr + j),
                    )

            x_gen_list.append(x_gen)

        if save_sampler_config:
            self.save(output_dir)

        if return_gen:
            return torch.cat(x_gen_list, dim=0)
        
if __name__ == '__main__':
    opts = TrainOptions().parse()
    
    vae_net = VAE()
    vae_net.to(opts.device)
    snapshot = torch.load('./models/vae_fm_0.312.pt')
    vae_net.load_state_dict(snapshot['model_state_dict'])
    vae_net.eval()
    
    """
    wae_net = WAE(latent_dims=16)
    wae_net.to(opts.device)
    snapshot = torch.load('./models/wae_m_paper.pt')
    wae_net.load_state_dict(snapshot['model_state_dict'])
    wae_net.eval()
    encoder = wae_net.encoder
    decoder = wae_net.decoder
    encoder.to(opts.device)
    decoder.to(opts.device)
    """
    """
    encoder = Encoder()
    decoder = Decoder()
    encoder.to(opts.device)
    decoder.to(opts.device)
    snapshot = torch.load('./models/rae_l2_m_paper.pt')
    encoder.load_state_dict(snapshot['model_state_dict']['encoder'])
    decoder.load_state_dict(snapshot['model_state_dict']['decoder'])
    encoder.eval()
    decoder.eval()
    """
    
    
    gmm_sampler = GaussianMixtureSampler(
        model=vae_net,
        encoder = vae_net.encoder,
        decoder = vae_net.decoder,
        n_components = 10
    )
    """
    
    gmm_sampler = GaussianMixtureSampler(
        encoder = encoder,
        decoder = decoder,
        n_components = 10
    )
    """
    """
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])   
    train = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=True)
    """
    
    
    transform = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])      
    train = datasets.FashionMNIST(root="./data/", transform=transform, train=True, download=True)    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=True)
    
    
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(32)])      
    train = datasets.CIFAR10(root="./data/", transform=transform, train=True, download=True)    
    data_train, data_validation = torch.utils.data.random_split(train, [len(train) - 10000, 10000], generator=torch.Generator().manual_seed(1))
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=100, shuffle=True)   
    """
    
    gmm_sampler.fit(data_train,opts,isVAE=True)
    
    gen_data = gmm_sampler.sample(
        num_samples=10000
    )
    
    with torch.no_grad():
                save_batches_of_images.save_set_of_images(os.path.join(opts.det, 'GMM'), gen_data)
    
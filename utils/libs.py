# -*- coding: utf-8 -*-


"""
    Ex-Post Density Estimation (XPDE).
"""
import torch
import numpy as np


class LatentSpaceSampler(object):
    def __init__(self, encoder, compute_z_cov=None):
        self.encoder = encoder
        
        

        self.z_cov = None

    def get_z_covariance(self, batches_of_xs):
        """Takes one or more batches of xs of shape batches X data_dims"""
        if isinstance(self.encoder(batches_of_xs), tuple) and len(self.encoder(batches_of_xs)) > 1:
            # Uncomment the following line if you want to fit multivariate gaussian for sampling for normal vaes too
            #zs = self.encoder(batches_of_xs)[0].detach().cpu().numpy()

            # Comment the following lines to not use unit variance for sampling all the time for normal VAE
            dim_z = self.encoder(batches_of_xs)[0].size(-1)

            z_origina_shape = (batches_of_xs.shape[0], dim_z)
            
            return np.eye(dim_z), z_origina_shape
        else:
            zs = self.encoder(batches_of_xs).detach().cpu().numpy()
        #zs = self.encoder(batches_of_xs).detach().cpu().numpy()

        z_original_shape = zs.shape
        zs = np.reshape(zs, (z_original_shape[0], -1))
        
        self.z_cov = np.cov(zs.T)  # https://blog.csdn.net/jeffery0207/article/details/83032325
        # shape of self.z_cov: [bottleneck_factor=16, bottleneck_factor=16]

        return self.z_cov, z_original_shape

    def get_zs(self, batches_of_xs):
        """batches_of_xs are only used to compute variance of Z on the fly"""
        num_smpls = batches_of_xs.shape[0]
        self.z_cov, z_dim = self.get_z_covariance(batches_of_xs)

        try:
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ), cov=self.z_cov,
                                                         size=num_smpls)  # https://www.zhihu.com/question/288946037/answer/649328934
        except np.linalg.LinAlgError as e:
            print(self.z_cov)
            print(e)
            zs_flattened = np.random.multivariate_normal(np.zeros(np.prod(z_dim[1:]), ),
                                                         cov=self.z_cov + 1e-5 * np.eye(self.z_cov.shape[0]),
                                                         size=num_smpls)

        return np.reshape(zs_flattened, (num_smpls,) + z_dim[1:])

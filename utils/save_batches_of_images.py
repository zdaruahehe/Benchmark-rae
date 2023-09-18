import os
from imageio import imwrite
import numpy as np
import matplotlib.pyplot as plt


def save_set_of_images(path, images):
    if not os.path.exists(path):
        os.mkdir(path)
    
    imgs = (np.clip(images.cpu().numpy(), 0, 1) * 255).astype('uint8')

    for i, img in enumerate(imgs):
        img = img.transpose(1, 2, 0)
        
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        imwrite(os.path.join(path, '%08d.png' % i), img)
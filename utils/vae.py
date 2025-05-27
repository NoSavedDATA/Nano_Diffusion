import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms

import numpy as np

import nosaveddata as nsd

from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionPipeline



class Vae:
    def __init__(self):

        model = "CompVis/stable-diffusion-v1-4"
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
        # pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

        

    def __call__(self, x):
        with torch.no_grad():
            latent_dist = self.vae.encode(x)
            latent = latent_dist.latent_dist.sample() 
            latent = 0.18215 * latent

        # print(f"z is: {latent.shape}")
        return latent

    def decode(self, latent):
        with torch.no_grad():
            decoded = self.vae.decode(latent / 0.18215)  
            decoded_image = decoded.sample  

        tensor = decoded_image.squeeze(0).detach().cpu()
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)  

        return tensor

    def decode_save(self, z, fname='out_imgs/test.png'):
        recon = self.decode(z)
        recon = transforms.ToPILImage()(recon)
        recon.save(fname)
        return recon




        



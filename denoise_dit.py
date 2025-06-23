import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import nosaveddata as nsd

from diffusers import StableDiffusionPipeline
from diffusers import DDPMScheduler, DDIMPipeline, DDIMScheduler, DDIMPipeline, DPMSolverMultistepScheduler

from utils.vae import Vae
from loader.dif_loader import Loader

import tqdm

from models.diffusion_model import DiffusionModel

import wandb, argparse
from diffusers.utils import make_image_grid

from PIL import Image




def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def denoise(unet, vae, step, config):
    scheduler = unet.noise_scheduler
    
    
    with torch.no_grad():
        z = torch.randn(16, 4, 32, 32, device='cuda')
        
        scheduler.set_timesteps(config.denoise.timesteps)
        
        timesteps = scheduler.timesteps
        
        for t in timesteps:
            z  = scheduler.scale_model_input(z, t)
            model_output = unet(z, torch.tensor([t]*16, device='cuda'))#.sample

            z = scheduler.step(
                    model_output, t, z,# eta=1, use_clipped_model_output=False,
                ).prev_sample


        image = vae.decode(z)
        
        image = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).numpy())
        # image_grid = make_image_grid(image, rows=4, cols=4) 
        # image_grid.save(f"out_imgs/{step:04d}.png")

        for i in range(16):
            
            image[i].save(f"unet_output_images/{step+i}.png")
            




if __name__=="__main__":
    print("Starting")


    config = nsd.read_yaml("configs/unet.yaml")


    vae = Vae()

    

    
    model = DiffusionModel(config).cuda()


    ckpt = torch.load('checkpoints/unet_dit.ckpt')
    model.load_state_dict(ckpt['model'])



    for i in tqdm.tqdm(range(10000)[::16]):
        denoise(model, vae, i, config)

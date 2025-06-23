import torch
from torch import nn
import torch.nn.functional as F

import torchvision

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


            


def denoise(pipe, step, prompt="placeholder text",
            output_path = "out_imgs/grid.png",
            num_images = 16, guidance_scale = 7.5,
            height = 256, width = 256, seed = 42):

    with torch.no_grad():
         
        

        images = pipe(
            [prompt] * num_images,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=200,
        ).images

        image_tensors = [torchvision.transforms.ToTensor()(img) for img in images]
        # grid = make_grid(torch.stack(image_tensors), nrow=4, padding=2)


        for i in range(num_images):
            images[i].save(f"unet_output_images/{step+i}.png")
            
        
        # output_path = f"out_imgs/{step:04d}.png"
        # save_image(grid, output_path)
        # print(f"Saved 4x4 image grid to {output_path}")



if __name__=="__main__":
    print("Starting")


    
    model_path = "checkpoints/sd15"

    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = None


    for i in tqdm.tqdm(range(10000)[::16]):
        denoise(pipe, i)

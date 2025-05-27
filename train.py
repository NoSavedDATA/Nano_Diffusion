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
    inference_timesteps = config.denoise.timesteps
    
    with torch.no_grad():
        z = torch.randn(16, 4, 32, 32, device='cuda')
        
        scheduler.set_timesteps(inference_timesteps)
        
        # timesteps = torch.arange(inference_timesteps)*(1000//inference_timesteps)
        # timesteps = timesteps.flip(0)[:-1]
        timesteps = scheduler.timesteps
        # print(f"{timesteps}")
        
        for t in timesteps:
            # 1. predict noise model_output
            z  = scheduler.scale_model_input(z, t)
            model_output = unet(z, torch.tensor([t]*16, device='cuda'))#.sample

            # 2. compute previous z: x_t -> x_t-1
            #z = scheduler.step(model_output, t, z).prev_sample
            z = scheduler.step(
                    model_output, t, z,# eta=1, use_clipped_model_output=False,
                ).prev_sample


        # print(f"DENOIZED Z {z.shape}")        
        # image = vae.decode_save(z[0][None], 'out_imgs/denoised.png')
        image = vae.decode(z)
        # print(f"DECODED {image.shape}")
        
        image = numpy_to_pil(image.cpu().permute(0, 2, 3, 1).numpy())    
        image_grid = make_image_grid(image, rows=4, cols=4)
     
        # Save the images   
        image_grid.save(f"out_imgs/{step:04d}.png")



mse = nn.MSELoss(reduction='none')
scaler = torch.cuda.amp.GradScaler()
def train_step(model, vae, optim, sched, batch, step, config, do_log):

    T_trunc = config.denoise.T_max
    x = batch
    x = x.cuda()


    z = vae(x)

    t = torch.randint(0, T_trunc-1, (x.shape[0],), device='cuda')
        
    '''Diffusion loss'''
    # with torch.autocast(device_type='cuda', dtype=torch.float16):
    noise = torch.randn_like(z)
    noised_z = model.noise_scheduler.add_noise(z, noise, t)
    
    y = model(noised_z, t)

    
    # loss = mse(y, noise).sum((1,3)).mean()
    loss = mse(y, noise).mean()
    # print(f"{loss.shape}")
    loss.backward()
    
    # scaler.scale(loss).backward()
    
    
    
    if (step+1)%config.train.acc_steps==0:
        # scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        # scaler.step(optim)
        # scaler.update()

        optim.step()
        optim.zero_grad()


        sched.step()


    # print(f"{loss}")

    if do_log:
        wandb.log({'loss': loss.cpu()})





if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    config = nsd.read_yaml("configs/unet.yaml")

    if args.log:
        wandb.init(
                project="Diffusion",
                name=f"Classifier Free Guidance",
                #id='ca225f0w',
                #resume='must',
                reinit=False
        )
        
    


    print(f"{config}")

    vae = Vae()    

    loader = Loader(config)

    
    model = DiffusionModel(config).cuda()
    optim = nsd.AdamW_wd(model, lr=config.train.lr, betas=(0.9,0.95))
    sched = nsd.WarmUp_Cosine(optim, config.train.warmup, config.train.lr, config.train.lr,
                              config.train.steps-config.train.warmup)

    step = 0
    max_step = config.train.steps * config.train.acc_steps

    if args.load:
        ckpt = torch.load(config.train.ckpt)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        sched.load_state_dict(ckpt['sched'])
        scaler.load_state_dict(ckpt['scaler'])
        step = ckpt['step']


    while step<max_step:
        for batch in tqdm.tqdm(loader()):

            train_step(model, vae, optim, sched, batch, step, config, args.log)
            step+=1

            if step%(config.train.save_every*config.train.acc_steps)==0:
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step}, config.train.ckpt)
                # model.noise_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, use_karras_sigmas=False, solver_order=2)
                denoise(model, vae, step, config)
                # model.noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012)



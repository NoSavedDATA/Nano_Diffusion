import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


from torchvision import transforms
from torchvision.utils import make_grid, save_image
import torchvision

import os
from PIL import Image

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL

from transformers import CLIPTokenizer, CLIPTextModel
from loader.games_loader import Loader

from diffusers.utils import make_image_grid

import nosaveddata as nsd

import tqdm, wandb, argparse



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


def denoise(model_path, step, prompt="placeholder text",
            output_path = "out_imgs/grid.png",
            num_images = 4, guidance_scale = 7.5,
            height = 256, width = 256, seed = 42):

    with torch.no_grad():
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
        pipe.safety_checker = None 
        generator = torch.Generator("cuda").manual_seed(seed)

        images = pipe(
            [prompt] * num_images,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=200,
        ).images

        image_tensors = [torchvision.transforms.ToTensor()(img) for img in images]
        grid = make_grid(torch.stack(image_tensors), nrow=4, padding=2)

        
        output_path = f"out_imgs/{step:04d}.png"
        save_image(grid, output_path)
        print(f"Saved 4x4 image grid to {output_path}")



def train_step(model, batch, optim, loader, vae, noise_scheduler, text_encoder, tokenizer, step, args, config):

    txt = "placeholder text"
    tokenized = tokenizer(txt, truncation=True, padding="max_length", return_tensors="pt")
    tokenized = tokenized.input_ids.cuda()


    x = batch.cuda()


    with torch.no_grad():
        tokenized = tokenized.repeat_interleave(x.shape[0], 0)

        latents = vae.encode(x).latent_dist.sample() * 0.18215

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device='cuda').long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = text_encoder(tokenized.cuda())[0]



    model_pred = model(noisy_latents, timesteps, encoder_hidden_states).sample
    loss = F.mse_loss(model_pred, noise)

    loss.backward()


    if args.log:
        wandb.log({"loss": loss})

    if (step+1)%config.train.acc_steps==0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()




def train(model, optim, loader, vae, noise_scheduler, text_encoder, tokenizer, step, args, config):
    




    max_step = config.train.steps * config.train.acc_steps
    while step < max_step:

        for batch in tqdm.tqdm(loader()):
            if step>=max_step:
                break


            train_step(model, batch, optim, loader, vae, noise_scheduler, text_encoder, tokenizer, step, args, config)
            step += 1


            
            if step%config.train.save_every==0:
                torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                            "step": step}, config.train.ckpt)

                pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    unet=model,                        
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    vae=vae
                )
                pipeline.save_pretrained("checkpoints/sd15") 
                
                denoise("checkpoints/sd15", step)



        model.save_pretrained(os.path.join(output_dir, f"unet-epoch-{epoch}"))







if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--load", action="store_true")
    args = parser.parse_args()

    if args.log:
        wandb.init(
                project="Diffusion",
                name=f"Classifier Free Guidance",
                #id='ca225f0w',
                #resume='must',
                reinit=False
        )


    config = nsd.read_yaml("configs/unet.yaml")


    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    instance_data_dir = "/path/to/data/instance_images"
    output_dir = "./dreambooth-sd-output"
    instance_prompt = "a photo of sks dog"


    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").cuda()
    model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").cuda()

    
    loader = Loader(config)

    
    optim = torch.optim.AdamW(model.parameters(), lr=5e-6, betas=(0.9,0.95))

    
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    
    step = 0
    
    if args.load:
        ckpt = torch.load(config.train.ckpt)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        step = ckpt['step']
        print(f"\nRecovering from step {step}.\n")




    train(model, optim, loader, vae, noise_scheduler, text_encoder, tokenizer, step, args, config)



    # Save results
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        unet=model,                        
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae
    )
    pipeline.save_pretrained("checkpoints/sd15") 

    model.save_pretrained(os.path.join(output_dir, f"unet-epoch-{epoch}"))
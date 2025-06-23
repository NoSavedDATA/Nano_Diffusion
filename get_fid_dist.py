import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import numpy as np
import os
from scipy import linalg

import tqdm

# from loader.games_loader import Loader
from loader.dit_gen_eval_loader import Loader
import nosaveddata as nsd


def get_inception_activations(loader, device):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # Output 2048D
    model.eval()

    activations = []

    with torch.no_grad():
        for images in tqdm.tqdm(loader()):
            images = images.to(device)
            if images.shape[2] != 299:
                images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear')
            act = model(images)
            activations.append(act.cpu().numpy())

    return np.concatenate(activations, axis=0)

def compute_stats(acts):
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma

def save_stats(mu, sigma, path):
    np.savez(path, mu=mu, sigma=sigma)

def process_and_save(loader, save_path):
    
    device = torch.device('cuda')
    activations = get_inception_activations(loader, device)
    mu, sigma = compute_stats(activations)
    save_stats(mu, sigma, save_path)

if __name__=="__main__":

    config = nsd.read_yaml("configs/unet.yaml")

    loader = Loader(config)
    process_and_save(loader, "fid_dists/sd15_finetune.npz")



    

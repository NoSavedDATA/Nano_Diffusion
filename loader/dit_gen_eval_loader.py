import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from torch.utils.data import DataLoader, Dataset


import os, glob

# preprocess = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),  # [0,1]
#     transforms.Normalize([0.5], [0.5])  # [-1,1]
# ])


preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


class DiffusionDataset(Dataset):
    def __init__(self, image_files):
        super().__init__()
        
        self.image_files = image_files
        print(f"DATASET HAVE {len(self.image_files)}")

    
    def __getitem__(self, idx):

        img = Image.open(self.image_files[idx])

        x = preprocess(img)

        return x

    def __len__(self):
        return len(self.image_files)
        


class Loader:
    def __init__(self, config):
        self.bs = config.train.bs
        self.num_workers = config.train.num_workers



        files = glob.glob("unet_output_images/*.png")

        print(f"GOT {len(files)} FILES")
        self.ds = DiffusionDataset(files)

    
    def __call__(self):
        # return DataLoader(self.ds, batch_size=self.bs, num_workers=self.num_workers, drop_last=True, shuffle=True)
        return DataLoader(self.ds, batch_size=self.bs, drop_last=True, shuffle=True)


    

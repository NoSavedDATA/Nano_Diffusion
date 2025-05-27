import torch
from torch import nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

from torch.utils.data import DataLoader, Dataset


import os, glob

# image = Image.open("your_image.png").convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize([0.5], [0.5])  # [-1,1]
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


        all_files = []
        # for i in range(153,169):
        for i in range(1,999):
            files = glob.glob(f"D:/datasets/ImageNet/train/{i}/*.JPEG")
            all_files.extend(files)
        files = all_files

        print(f"GOT {len(files)} FILES")
        self.ds = DiffusionDataset(files)

    
    def __call__(self):
        # return DataLoader(self.ds, batch_size=self.bs, num_workers=self.num_workers, drop_last=True, shuffle=True)
        return DataLoader(self.ds, batch_size=self.bs, drop_last=True, shuffle=True)


    

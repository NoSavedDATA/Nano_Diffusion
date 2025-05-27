import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from nosaveddata import *
import nosaveddata as nsd

from diffusers import UNet2DModel
from diffusers import DDPMScheduler, DDIMPipeline, DDIMScheduler, DDIMPipeline, DPMSolverMultistepScheduler




class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012)

        self.T_max = config.denoise.T_max

        hidden_groups = config.model.hidden_groups
        print(f"{hidden_groups}")
        print(f"{config.model.strides}")

        
        self.t_emb_dim = hidden_groups[0]
        self.t_emb_dim2 = hidden_groups[0]*4
        self.t_emb = nn.Sequential(nn.Linear(self.t_emb_dim, self.t_emb_dim2),
                                 nn.SiLU(),
                                 nn.Linear(self.t_emb_dim2, self.t_emb_dim2),
                                 nn.SiLU())
        # self.unet = nsd.UNet_Conditional(4, hidden_groups=hidden_groups, strides=config.model.strides, down_blocks=config.model.down_blocks,
        #                  up_blocks=config.model.up_blocks, num_blocks=config.model.num_blocks, t_emb=self.t_emb_dim2, c_emb_dim=self.t_emb_dim2,
        #                  res=(32,32), has_attn=False)

        # self.unet = UNet2DModel(
        #     sample_size=32,  # the target image resolution
        #     dropout=0.1,
        #     in_channels=4,  # the number of input channels, 3 for RGB images
        #     out_channels=4,  # the number of output channels
        #     layers_per_block=2,  # how many ResNet layers to use per UNet block
        #     block_out_channels=(256,512,512,512),  # the number of output channels for each UNet block
        #     down_block_types=(
        #         "DownBlock2D",  # a regular ResNet downsampling block
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         #"DownBlock2D",
        #         # "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        #         # "DownBlock2D",
        #     ),
        #     up_block_types=(
        #         # "UpBlock2D",  # a regular ResNet upsampling block
        #         # "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        #         #"UpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #     ),
        # ).cuda()

        self.unet = UNet_DiT(4, 768, 12, 768//64, patch=(2,2))
        
        
        
        """Init Weights"""
        self.init_weights()
    
        params_count(self, "Diffusion")
    
    
    def init_weights(self):
        self.t_emb.apply(init_relu)
        
    
    def forward(self, X, t, nlp_X=None, actions=None):
        # t_emb = sinusoidal_embedding(t, self.t_emb_dim)
        # t_emb = self.t_emb(t_emb)
        
        c_emb = torch.zeros(X.shape[0], 768, device='cuda')
        # c_emb = torch.zeros_like(t_emb)
        # print(f"{X.shape, t.shape}")

        # return self.unet(X, t_emb, c_emb)
        return self.unet(X, t, c_emb)
        # return self.unet(X, t).sample



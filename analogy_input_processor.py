"""
ADOBE CONFIDENTIAL
Copyright 2024 Adobe
All Rights Reserved.
NOTICE: All information contained herein is, and remains
the property of Adobe and its suppliers, if any. The intellectual
and technical concepts contained herein are proprietary to Adobe
and its suppliers and are protected by all applicable intellectual
property laws, including trade secret and copyright laws.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Adobe.
"""

import torch as th
from torchvision import transforms
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

DINO_SIZE = 224
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]

SIGLIP_SIZE = 256
SIGLIP_MEAN = [0.5]
SIGLIP_STD = [0.5]

    
class AnalogyInputProcessor(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(self,):
        super(AnalogyInputProcessor, self).__init__()
        
        self.dino_transform = transforms.Compose(
            [
                transforms.Resize((DINO_SIZE, DINO_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(DINO_MEAN, DINO_STD), 
            ]
        )
        
        self.siglip_transform = transforms.Compose(
            [
                transforms.Resize((SIGLIP_SIZE, SIGLIP_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(SIGLIP_MEAN, SIGLIP_STD), 
            ]
        )
        
        dino_mean = th.tensor(DINO_MEAN).view(1, 3, 1, 1)
        dino_std = th.tensor(DINO_STD).view(1, 3, 1, 1)
        siglip_mean = [SIGLIP_MEAN[0],] * 3
        siglip_std = [SIGLIP_STD[0],] * 3
        siglip_mean = th.tensor(siglip_mean).view(1, 3, 1, 1)
        siglip_std = th.tensor(siglip_std).view(1, 3, 1, 1)
        self.register_buffer("dino_mean", dino_mean)
        self.register_buffer("dino_std", dino_std)
        self.register_buffer("siglip_mean", siglip_mean)
        self.register_buffer("siglip_std", siglip_std)
        
    def __call__(self, analogy_prompt):
        # List of tuples of (A, A*, B)
        img_a_dino = []
        img_a_siglip = []
        img_a_star_dino = []
        img_a_star_siglip = []
        img_b_dino = []
        img_b_siglip = []
        
        for im_set in analogy_prompt:
            img_a, img_a_star, img_b = im_set
            img_a_dino.append(self.dino_transform(img_a))
            img_a_siglip.append(self.siglip_transform(img_a))
            img_a_star_dino.append(self.dino_transform(img_a_star))
            img_a_star_siglip.append(self.siglip_transform(img_a_star))
            img_b_dino.append(self.dino_transform(img_b))
            img_b_siglip.append(self.siglip_transform(img_b))
        
        img_a_dino = th.stack(img_a_dino, 0)
        img_a_siglip = th.stack(img_a_siglip, 0)
        img_a_star_dino = th.stack(img_a_star_dino, 0)
        img_a_star_siglip = th.stack(img_a_star_siglip, 0)
        img_b_dino = th.stack(img_b_dino, 0)
        img_b_siglip = th.stack(img_b_siglip, 0)
        
        dino_combined_input = th.stack([img_b_dino, img_a_dino, img_a_star_dino], 0)
        siglip_combined_input = th.stack([img_b_siglip, img_a_siglip, img_a_star_siglip], 0)
        
        return dino_combined_input, siglip_combined_input
    def get_negative(self, dino_in, siglip_in):
        
        dino_i = ((dino_in * 0 + 0.5) - self.dino_mean) / self.dino_std
        siglip_i = ((siglip_in * 0 + 0.5) - self.siglip_mean) / self.siglip_std
        return dino_i, siglip_i
            

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
from diffusers import ModelMixin
from transformers import AutoModel, SiglipVisionConfig, Dinov2Config
from transformers import SiglipVisionModel

from diffusers.configuration_utils import ConfigMixin, register_to_config
    
class AnalogyEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, load_pretrained=False, 
                 dino_config_dict=None, siglip_config_dict=None):
        super().__init__()
        if load_pretrained:
            image_encoder_dino = AutoModel.from_pretrained('facebook/dinov2-large', torch_dtype=th.float16)
            image_encoder_siglip = SiglipVisionModel.from_pretrained("google/siglip-large-patch16-256", torch_dtype=th.float16, attn_implementation="sdpa")
        else:
            image_encoder_dino = AutoModel.from_config(Dinov2Config.from_dict(dino_config_dict))
            image_encoder_siglip = AutoModel.from_config(SiglipVisionConfig.from_dict(siglip_config_dict))
            
        image_encoder_dino.requires_grad_(False)
        image_encoder_dino = image_encoder_dino.to(memory_format=th.channels_last)

        image_encoder_siglip.requires_grad_(False)
        image_encoder_siglip = image_encoder_siglip.to(memory_format=th.channels_last)
        self.image_encoder_dino = image_encoder_dino
        self.image_encoder_siglip = image_encoder_siglip


    def dino_normalization(self, encoder_output):
        embeds = encoder_output.last_hidden_state
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / th.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds
    
    def siglip_normalization(self, encoder_output):
        embeds = th.cat ([encoder_output.pooler_output[:, None, :], encoder_output.last_hidden_state], dim=1)
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / th.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds
    
    def forward(self, dino_in, siglip_in):

        x_1 = self.image_encoder_dino(dino_in, output_hidden_states=True)
        x_1_first = x_1.hidden_states[0]
        x_1 = self.dino_normalization(x_1)
        x_2 = self.image_encoder_siglip(siglip_in, output_hidden_states=True)
        x_2_first = x_2.hidden_states[0]
        x_2_first_pool = th.mean(x_2_first, dim=1, keepdim=True)
        x_2_first = th.cat([x_2_first_pool, x_2_first], 1)
        x_2 = self.siglip_normalization(x_2)
        dino_embd = th.cat([x_1, x_1_first], -1)
        siglip_embd = th.cat([x_2, x_2_first], -1)
        return dino_embd, siglip_embd
    
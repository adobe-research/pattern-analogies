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

import einops
import numpy as np
import torch as th
import torch.nn as nn
from diffusers import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
# REf: https://github.com/tatp22/multidim-positional-encoding/tree/master


OUT_SIZE = 768
IN_SIZE = 2048


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = th.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return th.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (th.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = th.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = th.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = th.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc



class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (th.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = th.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = th.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_z = th.arange(z, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = th.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = th.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = th.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = th.zeros(
            (x, y, z, self.channels * 3),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc

class AnalogyProjector(ModelMixin, ConfigMixin):
    
    @register_to_config
    def __init__(self):
        super(AnalogyProjector, self).__init__()
        self.projector = DinoSiglipMixer()
        self.pos_embd_1D = PositionalEncoding1D(OUT_SIZE)
        self.pos_embd_3D = PositionalEncoding3D(OUT_SIZE)
        

    def forward(self, dino_in, siglip_in, batch_size):
        
        image_embeddings = self.projector(dino_in, siglip_in)
        
        image_embeddings = einops.rearrange(image_embeddings, '(k b) t d -> b k t d', b=batch_size)
        image_embeddings = self.position_embd(image_embeddings)
        return image_embeddings

    def position_embd(self, image_embeddings, concat=False):
        canvas_embd = image_embeddings[:, :, 1:, :]
        batch_size = canvas_embd.shape[0]
        type_size = canvas_embd.shape[1]
        xy_size = canvas_embd.shape[2]
        
        x_size = int(xy_size ** 0.5)

        canvas_embd = canvas_embd.reshape(batch_size, type_size, x_size, x_size, -1)
        if concat:
            canvas_embd = th.cat([canvas_embd, self.pos_embd_3D(canvas_embd)], -1)
        else:
            canvas_embd = self.pos_embd_3D(canvas_embd) + canvas_embd
        canvas_embd = canvas_embd.reshape(batch_size, type_size, xy_size, -1)

        class_embd = image_embeddings[:, :, 0, :]
        if concat:
            class_embd = th.cat([class_embd, self.pos_embd_1D(class_embd)], -1)
        else:    
            class_embd = self.pos_embd_1D(class_embd) + class_embd
        all_embd_list = []
        for i in range(type_size):
            all_embd_list.append(class_embd[:, i:i+1])
            all_embd_list.append(canvas_embd[:, i])
        image_embeddings = th.cat(all_embd_list, 1)
        return image_embeddings


class HighLowMixer(th.nn.Module):
    def __init__(self, in_size=IN_SIZE, out_size=OUT_SIZE):
        super().__init__()
        mid_size = (in_size + out_size) // 2
        
        self.lower_projector = th.nn.Sequential(
            th.nn.LayerNorm(IN_SIZE//2),
            th.nn.SiLU()
        )
        self.upper_projector = th.nn.Sequential(
            th.nn.LayerNorm(IN_SIZE//2),
            th.nn.SiLU()
        )
        self.projectors = th.nn.ModuleList([
            # add layer norm
            th.nn.Linear(in_size, mid_size),
            th.nn.SiLU(),
            th.nn.Linear(mid_size, out_size)
        ])
        # initialize
        for proj in self.projectors:
            if isinstance(proj, th.nn.Linear):
                th.nn.init.xavier_uniform_(proj.weight)
                th.nn.init.zeros_(proj.bias)

    def forward(self, lower_in, upper_in, ):
        # ALso format lower_in
        lower_in = self.lower_projector(lower_in)
        upper_in = self.upper_projector(upper_in)
        x = th.cat([lower_in, upper_in], -1)
        for proj in self.projectors:
            x = proj(x)
        return x

class DinoSiglipMixer(th.nn.Module):
    def __init__(self, in_size=OUT_SIZE * 2, out_size=OUT_SIZE):
        super().__init__()
        self.dino_projector = HighLowMixer()
        self.siglip_projector = HighLowMixer()
        self.projectors = th.nn.Sequential(
            th.nn.SiLU(),
            th.nn.Linear(in_size, out_size),
        )
        # initialize
        for proj in self.projectors:
            if isinstance(proj, th.nn.Linear):
                th.nn.init.xavier_uniform_(proj.weight)
                th.nn.init.zeros_(proj.bias)

    
    def forward(self, dino_in, siglip_in):
        # ALso format lower_in
        lower, upper = th.chunk(dino_in, 2, -1)
        dino_out = self.dino_projector(lower, upper)
        lower, upper = th.chunk(siglip_in, 2, -1)
        siglip_out = self.siglip_projector(lower, upper)
        x = th.cat([dino_out, siglip_out], -1)
        for proj in self.projectors:
            x = proj(x)
        return x

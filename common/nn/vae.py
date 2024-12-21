import torch
from torch import nn
from diffusers import AutoencoderTiny

import torch
from torch import nn
import torch.nn.functional as F
import einops as eo
import numpy as np

from safetensors.torch import load_file

# haar filterbank
def get_haar_fb(device, dtype, ch_in = 3, invert = False):
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2, device = device, dtype = dtype)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2, device = device, dtype = dtype)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    if invert:
        haar_wav_lh *= -1
        haar_wav_hl *= -1

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh

def wavelet_tform(x):
    # [4,1,2,2]
    filters = torch.stack(get_haar_fb(x.device, x.dtype)).unsqueeze(1)

    x = F.interpolate(x, scale_factor=2,mode='bilinear')
    x = eo.rearrange(x, 'b c h w -> (b c) 1 h w')
    x = F.pad(x, (0,1,0,1), mode='constant', value=0)
    x = F.conv2d(x, filters, stride=2, padding=0)
    x = eo.rearrange(x, '(b c) d h w -> b (c d) h w', c = 3)

    return x
    
def conv(fi, fo):
    return nn.Conv2d(fi, fo, 3, 1, 1)

def proj(fi, fo, mode = 'same'):
    if mode == "up":
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(fi, fo, 3, 1, 1, bias = False)
        )
    return nn.Conv2d(fi, fo, 3, 1 if mode == 'same' else 2, 1, bias = False)

def up_proj(fi=64, fo=64):
    return proj(fi, fo, mode='up')

def down_proj(fi=64, fo=64):
    return proj(fi, fo, mode='down')

class GroupNorm(nn.GroupNorm):
    def __init__(self, n_groups, in_ch):
        super().__init__(n_groups, in_ch, eps = 1.0e-6, affine = True)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

# From TAESD
class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.gn1 = GroupNorm(32, 64)

        nn.init.normal_(self.conv2.weight, std=0.0001 / 64)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        residual = x.clone()

        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = F.silu(x)
        x = self.conv2(x)
        
        return x + residual
    
class TinyEncoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.conv_in = proj(12, 64)
        self.block_in = TinyBlock()
        
        self.down_blocks = nn.ModuleList([
            self._make_layer(1) for _ in range(5)
        ])
        
        self.conv_out = proj(64, latent_channels)
        
    def _make_layer(self, n_blocks):
        layers = []
        layers.append(down_proj())
        for _ in range(n_blocks):
            layers.append(TinyBlock())
        return nn.Sequential(*layers)
        
    def forward(self, x, output_hiddens=False):
        x = wavelet_tform(x)
        x = self.conv_in(x)
        x = self.block_in(x)
        
        h = []
        for block in self.down_blocks:
            x = block(x)
            h.append(x)
            
        x = self.conv_out(x)
        
        if output_hiddens:
            return x, h
        return x

class TinyDecoder(nn.Module):
    def __init__(self, latent_channels=64):
        super().__init__()
        self.clamp = Clamp()
        self.conv_in = proj(latent_channels, 64)
        self.relu = nn.ReLU()
        
        self.up_blocks = nn.ModuleList([
            self._make_layer(1) for _ in range(5)
        ])
        
        self.block_out = TinyBlock()
        self.conv_out = proj(64, 3)
        
    def _make_layer(self, n_blocks):
        layers = []
        for _ in range(n_blocks):
            layers.append(TinyBlock())
        layers.append(up_proj())
        return nn.Sequential(*layers)
        
    def forward(self, x, output_hiddens=False):
        x = self.clamp(x)
        x = self.conv_in(x)
        x = self.relu(x)
        
        h = []
        for block in self.up_blocks:
            x = block(x)
            h.append(x)
            
        x = self.block_out(x)
        x = self.conv_out(x)
        
        if output_hiddens:
            return x, h
        return x

def load_enc_dec(path):
    from safetensors.torch import load_file
    
    # Load checkpoint
    ckpt = load_file(path)
    
    # Initialize empty dicts for each model's weights
    encoder_ckpt = {}
    decoder_ckpt = {}
    
    # Sort weights into encoder and decoder dicts
    for key in ckpt.keys():
        if key.startswith('encoder.'):
            encoder_ckpt[key[8:]] = ckpt[key]  # Remove 'encoder.' prefix
        elif key.startswith('decoder.'):
            decoder_ckpt[key[8:]] = ckpt[key]  # Remove 'decoder.' prefix
            
    # Initialize models
    encoder = TinyEncoder(64)
    decoder = TinyDecoder(64)
    
    # Load weights
    encoder.load_state_dict(encoder_ckpt)
    decoder.load_state_dict(decoder_ckpt)
    
    return encoder, decoder

class VAE(nn.Module):
    def __init__(self, path = None, force_batch_size = 64):
        super().__init__()

        self.encoder, self.decoder = load_enc_dec("vae.safetensors")
        self.encoder.cuda()
        self.decoder.cuda()
        self.encoder.to(torch.bfloat16)
        self.decoder.to(torch.bfloat16)
        
        self.force_batch_size = force_batch_size
        
    @torch.no_grad()
    def encode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            latents = [self.encoder(chunk.bfloat16()) for chunk in chunks]
            return torch.cat(latents, dim=0).to(x_dtype)
        return self.encoder(x.bfloat16()).to(x_dtype)
    
    @torch.no_grad()
    def decode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            decoded = [self.decoder(chunk.bfloat16()) for chunk in chunks]
            return torch.cat(decoded, dim=0).to(x_dtype)
        return self.decoder(x.bfloat16()).to(x_dtype)
    
    @torch.no_grad()
    def forward(self, latents):
        return self.decoder(latents)
        
class SDVAE(nn.Module):
    def __init__(self, path = "madebyollin/taesdxl", force_batch_size = 16):
        super().__init__()

        self.model = AutoencoderTiny.from_pretrained(path)#taef1", torch_dtype = torch.half)
        self.model.cuda()
        self.model.half()

        self.force_batch_size = force_batch_size
    
    @torch.no_grad()
    def encode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            latents = [self.model.encode(chunk.half()).latents for chunk in chunks]
            return torch.cat(latents, dim=0).to(x_dtype)
        return self.model.encode(x.half()).latents.to(x_dtype)
    
    @torch.no_grad()
    def decode(self, x):
        x_dtype = x.dtype
        if self.force_batch_size is not None:
            chunks = x.split(self.force_batch_size)
            decoded = [self.forward(chunk.half()) for chunk in chunks]
            return torch.cat(decoded, dim=0).to(x_dtype)
        return self.forward(x.half()).to(x_dtype)
    
    @torch.no_grad()
    def forward(self, latents):
        return self.model.decode(latents).sample
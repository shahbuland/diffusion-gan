from common.nn.embeddings import TimestepEmbedding
from common.nn.mlp import MLP
from common.nn.transformer import StackedDiT
from common.nn.text_embedder import TextEmbedder
from common.nn.vae import VAE
from common.nn.modulation import FinalMod

from common.configs import ModelConfig
from common.utils import freeze, unfreeze

import einops as eo
import torch
from torch import nn
import torch.nn.functional as F

class RFTCore(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.blocks = StackedDiT(config)

        self.t_embed = TimestepEmbedding(config.d_model)
        self.pool_embed = MLP(config.text_d_model, dim_out = config.d_model)
        self.text_proj = nn.Linear(config.text_d_model, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, config.channels, bias = False)
        self.final_mod = FinalMod(config.d_model)
        
        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            1, 1, 0, bias = False
        )

        self.depatchify = lambda  x: eo.rearrange(
            x,
            'b (n_y n_x) c -> b c n_y n_x',
            n_y = config.sample_size
        )

    def forward(self, x, y, ts):
        y_pool = y.clone().mean(1)
        y = self.text_proj(y)
        x = self.patch_proj(x).flatten(2).transpose(1,2)

        cond = self.t_embed(ts) + self.pool_embed(y_pool)
        x = self.blocks(x, y, cond)

        x = self.final_mod(x, cond)
        x = self.proj_out(x)
        x = self.depatchify(x)

        return x

class RFT(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.config = config
        self.core = RFTCore(config)

        self.text_embedder = TextEmbedder()
        self.vae = VAE()
        freeze(self.text_embedder)
        freeze(self.vae)
    
    def parameters(self, group=None):
        if group is None:
            return self.core.parameters()
        elif group == 'projection':
            return [
                *self.core.patch_proj.parameters(),
                *self.core.proj_out.parameters()
            ]
        elif group == 'main':
            return [
                *self.core.blocks.parameters(),
                *self.core.text_proj.parameters(),
                *self.core.pool_embed.parameters(), 
                *self.core.t_embed.parameters(),
                *self.core.final_mod.parameters()
            ]
        else:
            raise ValueError(f"Unknown parameter group: {group}")

    def encode_text(self, *args, **kwargs):
        return self.text_embedder.encode_text(*args, **kwargs)

    def forward(self, x):
        if self.config.take_label:
            x, ctx = x # c is list str
            if self.config.cfg_prob > 0:
                mask = torch.rand(len(ctx)) < self.config.cfg_prob
                ctx = [c if not m else "" for c, m in zip(ctx, mask)]

            ctx = self.text_embedder.encode_text(ctx)
            ctx = ctx.to(x.dtype).to(x.device)
        else:
            ctx = None
        
        with torch.no_grad():
            x = self.vae.encode(x)

        b,c,h,w = x.shape
        with torch.no_grad():
            z = torch.randn_like(x)
            t = torch.randn(b, device = x.device, dtype = x.dtype).sigmoid() # log norm timesteps
            
            t_exp = eo.repeat(t, 'b -> b c h w', c = c, h = h, w = w)
            lerpd = x * (1. - t_exp) + z * t_exp
            target = z - x
        
        extra = {}
        total_loss = 0.

        pred = self.core(lerpd, ctx, t)
        diff_loss = F.mse_loss(target, pred)
        total_loss += diff_loss
        extra['diff_loss'] = diff_loss.item()
        return total_loss, extra

    def denoise(self, *args, **kwargs):
        return self.core(*args, **kwargs)

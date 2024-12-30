from common.nn.embeddings import TimestepEmbedding, StepEmbedding
from common.nn.mlp import MLP
from common.nn.transformer import StackedDiT
from common.nn.text_embedder import TextEmbedder
from common.nn.vae import VAE
from common.nn.modulation import FinalMod

from common.configs import ModelConfig
from common.utils import freeze, unfreeze
from common.utils import sample_step_size, sample_discrete_timesteps

import einops as eo
import torch
from torch import nn
import torch.nn.functional as F

class RFTCore(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.blocks = StackedDiT(config)

        self.t_embed = TimestepEmbedding(config.d_model)
        self.d_embed = StepEmbedding(config.d_model)

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

    def forward(self, x, y, ts, d):
        y_pool = y.clone().mean(1)
        y = self.text_proj(y)
        x = self.patch_proj(x).flatten(2).transpose(1,2)

        cond = self.t_embed(ts) + self.pool_embed(y_pool) + self.d_embed(d)
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

        with torch.no_grad():
            self.empty_embed = self.text_embedder.encode_text([""])

        self.ema_denoise_fn = None

    def set_ema(self, ema):
        # Expects unwrapped ema_model
        self.ema_denoise_fn = ema.ema_model.denoise

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

    def get_sc_loss(self, x):
        # Generate targets without grad
        with torch.no_grad():
            if self.config.take_label:
                x, ctx = x
                ctx = self.text_embedder.encode_text(ctx)
                ctx = ctx.to(x.dtype).to(x.device)
                neg_ctx = self.empty_embed.repeat(x.shape[0],1,1).to(x.dtype).to(x.device)
            else:
                ctx = None

            if self.vae is not None:
                x = self.vae.encode(x)
            
            # Mostly the same, but we sample steps first then sample time based on those
            b,c,h,w = x.shape
            z = torch.randn_like(x)
            
            d_slow = sample_step_size(b, self.config.base_steps).to(device=x.device,dtype=x.dtype)
            cfg_mask = (d_slow == 128)[:,None,None,None]
            d_fast = d_slow / 2 # half as may steps -> faster
    
            dt_slow = -1./d_slow
            dt_fast = -1./d_fast

            t = sample_discrete_timesteps(d_fast)
            
            def expand(u):
                return eo.repeat(u, 'b -> b c h w', c = c, h = h, w = w)
            
            t_exp = expand(t)
            dt_exp = expand(dt_slow)

            noisy = x * (1. - t_exp) + z * t_exp
            pred_1 = self.ema_denoise_fn(noisy, ctx, t, d_slow)
            if cfg_mask.any():
                pred_1_neg = self.ema_denoise_fn(noisy, neg_ctx, t, d_slow)
                pred_1 = torch.where(
                    cfg_mask,
                    pred_1_neg + self.config.sc_cfg * (pred_1 - pred_1_neg),
                    pred_1
                )
            
            less_noisy = noisy + dt_exp * pred_1
            pred_2 = self.ema_denoise_fn(less_noisy, ctx, t+dt_slow, d_slow)
            if cfg_mask.any():
                pred_2_neg = self.ema_denoise_fn(less_noisy, neg_ctx, t+dt_slow, d_slow)
                pred_2 = torch.where(
                    cfg_mask,
                    pred_2_neg + self.config.sc_cfg * (pred_2 - pred_2_neg),
                    pred_2
                )
            
            sc_target = 0.5 * (pred_1 + pred_2)
        
        # Now we have target
        sc_pred = self.denoise(noisy, ctx, t, d_fast)
        sc_loss = F.mse_loss(sc_target, sc_pred)
        return sc_loss

    def forward(self, x):
        # Split batch into regular and shortcut samples based on sc_batch_frac
        batch_size = len(x[0])
        n_regular = int(batch_size * (1 - self.config.sc_batch_frac))
        x, sc_x = x[:n_regular], x[n_regular:]

        x, ctx = x
        x, sc_x = x[:n_regular], x[n_regular:]
        ctx, sc_ctx = ctx[:n_regular], ctx[n_regular:]

        x = (x, ctx)
        sc_x = (sc_x, sc_ctx)
        
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
            
            # 3 different ways to sample t (compare these)
            t = torch.randn(b, device = x.device, dtype = x.dtype).sigmoid() # log norm timesteps
            #t = torch.rand(b, device = x.device, dtype = x.dtype)
            #t = sample_discrete_timesteps(self.config.base_steps, device = x.device, dtype = x.dtype)
            d = torch.full_like(t, self.config.base_steps)

            t_exp = eo.repeat(t, 'b -> b c h w', c = c, h = h, w = w)
            lerpd = x * (1. - t_exp) + z * t_exp
            target = z - x
        
        extra = {}
        total_loss = 0.

        pred = self.core(lerpd, ctx, t, d)
        diff_loss = F.mse_loss(target, pred)
        total_loss += diff_loss

        if sc_x is not None:
            sc_loss = self.get_sc_loss(sc_x)
            total_loss += sc_loss
            extra['sc_loss'] = sc_loss

        extra['diff_loss'] = diff_loss.item()
        return total_loss, extra

    def denoise(self, *args, **kwargs):
        return self.core(*args, **kwargs)

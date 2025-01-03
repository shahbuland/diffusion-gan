import torch
from torch import nn
import torch.nn.functional as F
import einops as eo
import math

from rotary_embedding_torch import RotaryEmbedding

from .mlp import MLP

class RoPEEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.rope = RotaryEmbedding(dim//2, theta=100)

    def flip(self, x):
        return x.transpose(1,2) # (swap seq and heads or reverse)
    
    def forward(self, q, k):
        q = self.flip(q)
        k = self.flip(k)

        orig_dtype = q.dtype
        q = q.float()
        k = k.float()

        q,k = self.rope.rotate_queries_or_keys(q),self.rope.rotate_queries_or_keys(k)

        q = q.to(orig_dtype)
        k = k.to(orig_dtype)
        
        q = self.flip(q)
        k = self.flip(k)

        return q,k

class SinCosEmbedding(nn.Module):
    def __init__(self, d, theta=300, mult=1000):
        super().__init__()
        self.d = d # Assume this is even
        self.theta = theta
        self.mult = mult

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = t * self.mult
        half = self.d // 2

        inds = torch.arange(half, device=t.device, dtype=t.dtype)
        freqs = (
            -math.log(self.theta) * inds / half
        ).exp()

        embs = t[:,None] * freqs[None]
        embs = torch.cat([torch.cos(embs), torch.sin(embs)], dim=-1)
        return embs


class TimestepEmbedding(nn.Module):
    def __init__(self, d_out, d_in = 512, mult = 1000):
        super().__init__()

        self.mlp = MLP(d_in, dim_out=d_out)
        self.sincos = SinCosEmbedding(d_in, theta=300, mult=mult)

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.mlp.fc_uv.weight.device, dtype=self.mlp.fc_uv.weight.dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        assert torch.all((t >= 0) & (t <= 1)), f"Timesteps must be in [0,1], got {t.min():.3f} to {t.max():.3f}"

        embs = self.sincos(t)
        return self.mlp(embs)

class StepEmbedding(nn.Module):
    def __init__(self, d_out, d_in=512, max_steps=128):
        super().__init__()

        self.mlp = MLP(d_in, dim_out=d_out)
        self.max_steps = max_steps
        mult = 1000 / math.log2(max_steps)
        self.sincos = SinCosEmbedding(d_in, theta=300, mult=mult)

    def forward(self, steps):
        if not isinstance(steps, torch.Tensor):
            steps = torch.tensor(steps, device=self.mlp.fc_uv.weight.device, dtype=self.mlp.fc_uv.weight.dtype)
        if steps.ndim == 0:
            steps = steps.unsqueeze(0)

        # Map steps to [0, log2(max_steps)]
        t = (math.log2(self.max_steps) - torch.log2(steps.float())).to(steps.dtype)
        embs = self.sincos(t)
        return self.mlp(embs)


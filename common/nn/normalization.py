import torch
from torch import nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps = 1.0e-6):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x):
        # x is [b,n,d] or [b,n,h,d]
        if x.dim() == 4:
            gain = (1 + self.g)[None,None,None,:] # Add batch, sequence, and head dims
        else:
            gain = (1 + self.g)[None,None,:] # Add batch and sequence dims

        rms = (x.float().pow(2).mean(-1, keepdim = True) + self.eps).rsqrt() # [b, n]

        x = (x * rms.to(x.dtype))
        x = x * gain

        return x

class RMSNormOpt(nn.Module):
    def __init__(self, gain, eps = 1.0e-6):
        super().__init__()

        # Expand gain to [1,1,1,d] for broadcasting during inference
        self.g = nn.Parameter((1 + gain)[None,None,None,:])
        self.eps = eps

    def forward(self, x):
        # x is [b,h,n,d]
        rms = (x.float().pow(2).mean(-1, keepdim=True) + self.eps).rsqrt() # [b,h,n,1]
        return (x * rms.to(x.dtype)) * self.g

LayerNorm = lambda dim: nn.LayerNorm(dim, elementwise_affine = False, eps = 1.0e-6)

class GroupNorm(nn.GroupNorm):
    def __init__(self, n_groups, in_ch):
        super().__init__(n_groups, in_ch, eps = 1.0e-6, affine = False)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            None, None,
            self.eps,
        )
        return output.type_as(input)
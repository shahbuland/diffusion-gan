import torch
from torch import nn
import torch.nn.functional as F

from .modulation import DoubleModConv

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=512):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride = 1, padding=1)

        self.gn1 = GroupNorm(32, in_channels)
        self.mod1 = DoubleModConv(cond_dim, in_channels, out_channels)
        self.gn2 = GroupNorm(32, out_channels)
        self.mod2 = DoubleModConv(cond_dim, in_channels, out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
        nn.init.normal_(self.conv2.weight, std=0.0001 / out_channels)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, cond):
        residual = self.shortcut(x.clone())
        mod1 = self.mod1(cond)
        mod2 = self.mod2(cond)

        x = self.gn1(x)
        x = mod1.first_step(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = mod1.second_step(x)

        x = self.gn2(x)
        x = mod2.first_step(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = mod2.second_step(x)
        
        return x + residual

        
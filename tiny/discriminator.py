import torch
import torch.nn.functional as F
from torch import nn
import math
from torchvision import models

from common.nn.vae import VAE
from common.utils import freeze

class DeepDiscHead(nn.Module):
    def __init__(self, fi, k1, s1, k2, s2):
        super().__init__()

        fo = fi // 2
        self.conv1 = nn.Conv2d(fi, fo, k1, s1, 0)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(fo, 1, k2, s2, 0)

        nn.init.zeros_(self.conv2.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x.flatten(1) # [b, 1, h, w] -> [b, h*w]
        return x

class SimpleDiscHead(nn.Module):
    def __init__(self, fi, k, s):
        super().__init__()

        self.conv = nn.Conv2d(fi, 1, k, s, 0)
        nn.init.zeros_(self.conv.weight)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1) # [b, 1, h, w] -> [b, h*w]
        return x


class VGGDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.inds = [3, 8, 15, 22, 29]
        # block ch are 64,128,256,512,512
        # block dim are 256,128,,64,32,16

        slices = [self.vgg.features[0:self.inds[0]]]
        for i in range(len(self.inds)-1):
            slices.append(self.vgg.features[self.inds[i]:self.inds[i+1]])
        self.slices = nn.ModuleList(slices)

        self.heads = nn.ModuleList([
            DeepDiscHead(64, 4, 4, 4, 4),
            DeepDiscHead(128, 4, 4, 2, 2),
            DeepDiscHead(256, 2, 2, 2, 2),
            SimpleDiscHead(512, 2, 2),
            SimpleDiscHead(512, 1, 1)
        ])
        
    def net(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        xs = [x]
        hs = []
        for slice, head in zip(self.slices, self.heads):
            xs.append(slice(xs[-1]))
            hs.append(head(xs[-1].clone()))
        
        return sum(hs)
    
    def forward(self, x_fake, x_real = None):
        if x_real is None:
            return -self.net(x_fake).mean()
        else:
            fake_out = self.net(x_fake)
            real_out = self.net(x_real)

            loss_fake = F.relu(1 + fake_out).mean()
            loss_real = F.relu(1 - real_out).mean()
            return (loss_real + loss_fake) * 0.5


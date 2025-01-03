import torch
from torch import nn
from torch.nn import functional as F
import einops as eo

from .normalization import LayerNorm

class ModulationOutput:
    def __init__(self, alpha, beta, gamma, conv = False):
        def expand(x):
            if conv:
                x = x[:,None,None]
            else:
                x = x[:,None]
            return x
    
        self.alpha = expand(alpha)
        self.beta = expand(beta)
        self.gamma = expand(gamma)

    def first_step(self, x): # [b,n,d]
        return x * (1 + self.alpha) + self.beta
    
    def second_step(self, x):
        return x * self.gamma

class DoubleModBlock(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim

        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, 6 * dim_out)
    
    def forward(self, cond): # [b,d]
        cond = self.act(cond)
        params = self.fc(cond)
        alpha_1, beta_1, gamma_1, alpha_2, beta_2, gamma_2 = params.chunk(6, dim = -1) # Break into 6 parts
        return [
            ModulationOutput(alpha_1, beta_1, gamma_1, conv = False),
            ModulationOutput(alpha_2, beta_2, gamma_2, conv = False)
        ]

class DoubleModConv(nn.Module):
    def __init__(self, fi, fo, cond_dim):
        super().__init__()

        self.act = nn.SiLU()
        # First modulation operates on fi channels, second on fo channels
        self.fc = nn.Linear(cond_dim, 2 * fi + 4 * fo)
        self.fi = fi
        self.fo = fo
    
    def forward(self, cond): # [b,d]
        cond = self.act(cond)
        params = self.fc(cond)

        part_1 = params[:,:self.fi*2]
        part_2 = params[:,self.fi*2:]

        alpha_1, beta_1 = part_1.chunk(2, dim = -1)
        gamma_1, alpha_2, beta_2, gamma_2 = part_2.chunk(4, dim = -1)

        first_mod = ModulationOutput(alpha_1, beta_1, gamma_1, conv=True)
        second_mod = ModulationOutput(alpha_2, beta_2, gamma_2, conv=True)
        
        return [first_mod, second_mod]

class FinalMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.ln = LayerNorm(dim)

        self.act = nn.SiLU()
        self.fc = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond): #[b,n,d],[b,d]
        cond = self.act(cond)

        x = self.ln(x)
        scale, shift = self.fc(cond).chunk(2,dim=-1) # 2*[b,d]
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1) # Add n dim to both

        return x * (1. + scale) + shift

class FinalGNMod(nn.GroupNorm):
    def __init__(self, n_groups, in_ch, cond_dim):
        super().__init__(n_groups, in_ch, eps = 1.0e-6, affine = False)

        self.act = nn.SiLU()
        self.fc = nn.Linear(cond_dim, 2 * in_ch)

    def expand_to_conv(self, x):
        return eo.rearrange(x, 'b c -> b c 1 1')

    def forward(self, input, cond):
        cond = self.act(cond)
        scale, shift = self.fc(cond).chunk(2,dim=-1)
        scale = self.expand_to_conv(scale).float()
        shift = self.expand_to_conv(shift).float()

        output = F.group_norm(
            input.float(),
            self.num_groups,
            None, None, # weight and bias
            self.eps,
        )
        output = output * (1. + scale) + shift
        return output.type_as(input)


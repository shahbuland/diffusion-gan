from shortcut.models import RFTCore as TeacherRFTCore

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

class Teacher(TeacherRFTCore):
    def __init__(self, checkpoint_path, config):
        super().__init__(config)
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path)
        self.load_state_dict(state_dict)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x, ctx):
        # Project text embeddings
        y_pool = ctx.clone().mean(1)
        y = self.text_proj(y)
        
        # Project image patches
        x = self.patch_proj(x).flatten(2).transpose(1,2)
        
        # Fixed t=1, d=1 conditioning
        t = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        d = torch.ones_like(t)
        cond = self.t_embed(t) + self.pool_embed(y_pool) + self.d_embed(d)
        
        # Get transformer outputs with hidden states
        x, hidden_states = self.blocks(x, y, cond, output_hidden_states=True)
        
        # Final processing
        x = self.final_mod(x, cond)
        x = self.proj_out(x)
        x = self.depatchify(x)
        
        return x, hidden_states

class TinyRFTCore(nn.Module):
    def __init__(self, config : ModelConfig):
        super().__init__()

        self.blocks = StackedDiT(config)
        self.pool_embed = MLP(config.text_d_model, dim_out = config.d_model)
        self.text_proj = nn.Linear(config.text_d_model, config.d_model, bias = False)
        self.proj_out = nn.Linear(config.d_model, config.channels, bias = False)
        self.final_mod = FinalMod(config.d_model)
        
        self.patch_proj = nn.Conv2d(
            config.channels,
            config.d_model,
            1, 1, 0, bias = False
        )

        self.depatchify = lambda x: eo.rearrange(
            x,
            'b (n_y n_x) c -> b c n_y n_x',
            n_y = config.sample_size
        )

    def forward(self, x, y, output_hidden_states=False):
        y_pool = y.clone().mean(1)
        y = self.text_proj(y)
        x = self.patch_proj(x).flatten(2).transpose(1,2)

        cond = self.pool_embed(y_pool)
        x, hidden_states = self.blocks(x, y, cond, output_hidden_states=True)

        x = self.final_mod(x, cond)
        x = self.proj_out(x)
        x = self.depatchify(x)

        if output_hidden_states:
            return x, hidden_states
        return x

class TinyRFT(nn.Module):
    def __init__(self, config : ModelConfig, teacher_path : str):
        super().__init__()

        self.config = config
        self.core = TinyRFTCore(config)
        
        self.text_embedder = TextEmbedder()
        self.vae = VAE()
        freeze(self.text_embedder)
        freeze(self.vae)

        self.teacher = Teacher(teacher_path, config)
        freeze(self.teacher)

    def parameters(self):
        return self.core.parameters()

    def denoise(self, *args, **kwargs):
        return self.core(*args, **kwargs)

    def encode_text(self, *args, **kwargs):
        return self.text_embedder.encode_text(*args, **kwargs)

    def forward(self, x):
        if self.config.take_label:
            x, ctx = x
            ctx = self.text_embedder.encode_text(ctx)
            ctx = ctx.to(x.dtype).to(x.device)
        else:
            ctx = None
        
        with torch.no_grad():
            x = self.vae.encode(x)
            z = torch.randn_like(x)
            target = z - x
            
            # Get teacher outputs
            teacher_pred, teacher_hidden = self.teacher(x, ctx)

        # Get student prediction and hidden states
        pred, student_hidden = self.core(x, ctx, output_hidden_states=True)
        
        diff_loss = F.mse_loss(target, pred)
        
        extra = {
            'diff_loss': diff_loss.item()
        }
        
        return diff_loss, extra
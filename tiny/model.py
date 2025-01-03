from shortcut.model import RFTCore as TeacherRFTCore

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

        # Filter and clean state dict keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ema_model.core.'):
                new_key = key.replace('ema_model.core.', '')
                new_state_dict[new_key] = value
        
        self.load_state_dict(new_state_dict)
        
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad() 
    def forward(self, x, y):
        # Project text embeddings
        y_pool = y.clone().mean(1)
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

class FeatureMatchingLayer(nn.Module):
    def __init__(self, dim_student, dim_teacher):
        super().__init__()

        self.proj = (
            MLP(dim_student, dim_out=dim_teacher) 
            if dim_student != dim_teacher 
            else nn.Sequential()
        )
    
    def forward(self, h_student, h_teacher):
        # both [b,n,d]
        h_student = self.proj(h_student)
        h_student = F.normalize(h_student, dim = -1)
        h_teacher = F.normalize(h_teacher, dim = -1)
        cos_sims = (h_student * h_teacher).sum(-1) # [b,n]
        feature_loss = (-cos_sims).mean()
        return feature_loss

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, dim_student, dim_teacher, layer_inds_student, layer_inds_teacher):
        super().__init__()
        
        assert len(layer_inds_student) == len(layer_inds_teacher), \
            "Must provide same number of student and teacher layer indices"
            
        self.layer_inds_student = layer_inds_student
        self.layer_inds_teacher = layer_inds_teacher
        self.feature_matchers = nn.ModuleList([
            FeatureMatchingLayer(dim_student, dim_teacher)
            for _ in range(len(layer_inds_student))
        ])

    def forward(self, h_student_list, h_teacher_list):
        total_loss = 0.
        for (student_idx, teacher_idx), matcher in zip(
            zip(self.layer_inds_student, self.layer_inds_teacher), 
            self.feature_matchers
        ):
            h_student = h_student_list[student_idx]
            h_teacher = h_teacher_list[teacher_idx]
            total_loss += matcher(h_student, h_teacher)
        
        return total_loss / len(self.layer_inds_student)

class TinyRFT(nn.Module):
    def __init__(self, config : ModelConfig, teacher_config : ModelConfig, teacher_path : str):
        super().__init__()

        self.config = config
        self.core = TinyRFTCore(config)
        
        self.text_embedder = TextEmbedder()
        self.vae = VAE()
        freeze(self.text_embedder)
        freeze(self.vae)

        self.teacher = Teacher(teacher_path, teacher_config)
        freeze(self.teacher)

        self.kd_loss = KnowledgeDistillationLoss(
            config.d_model,
            teacher_config.d_model,
            config.kd_inds_student,
            config.kd_inds_teacher
        )

    def parameters(self):
        yield from self.core.parameters()
        yield from self.kd_loss.parameters()

    def denoise(self, *args, **kwargs):
        return self.core(*args, **kwargs)

    def encode_text(self, *args, **kwargs):
        return self.text_embedder.encode_text(*args, **kwargs)

    def forward(self, x):
        x, ctx = x
        ctx = ctx.unsqueeze(1).repeat(1, 77, 1).to(x.dtype)
        
        with torch.no_grad():
            z = torch.randn_like(x)
            target = z - x
            
            # Get teacher outputs
            teacher_pred, teacher_hidden = self.teacher(z, ctx)

        total_loss = 0.

        # Get student prediction and hidden states
        pred, student_hidden = self.core(z, ctx, output_hidden_states=True)
        
        diff_loss = F.mse_loss(target, pred)
        kd_loss = self.kd_loss(student_hidden, teacher_hidden)
        
        total_loss += diff_loss
        total_loss += kd_loss

        with torch.no_grad():
            orig_x = self.vae.decoder(x)

        extra = {
            'diff_loss': diff_loss.item(),
            'kd_loss': kd_loss.item(),
            'samples' : self.vae.decoder(z - pred),
            'original' : orig_x
        }
        
        return total_loss, extra
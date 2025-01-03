from shortcut.model import RFT, RFTCore
from tiny.model import KnowledgeDistillationLoss

from common.utils import freeze
import torch
from torch import nn
import einops as eo

import torch.nn.functional as F
from common.utils import sample_step_size, sample_discrete_timesteps

class TeacherStudentRFT(RFT):
    def __init__(self, student_config, teacher_config, teacher_ckpt):
        super().__init__(student_config)

        self.teacher = RFTCore(teacher_config)

        # Load teacher ckpt
        # Only get core, and get ema model
        state_dict = torch.load(teacher_ckpt)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ema_model.core.'):
                new_key = key.replace('ema_model.core.', '')
                new_state_dict[new_key] = value
        
        self.teacher.load_state_dict(new_state_dict)
        freeze(self.teacher)
    
        self.kd_loss = KnowledgeDistillationLoss(
            student_config.d_model,
            teacher_config.d_model,
            student_config.kd_inds_student,
            student_config.kd_inds_teacher
        )

    def get_sc_loss(self, x):
        # Generate targets without grad
        with torch.no_grad():
            x, ctx = x
            ctx = ctx.unsqueeze(1).repeat(1, 77, 1).to(x.dtype)
            neg_ctx = self.empty_embed.repeat(x.shape[0],1,1).to(x.dtype).to(x.device)

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

            with torch.no_grad():
                teacher_pred, teacher_h = self.teacher(noisy, ctx, t, d_fast, output_hidden_states = True)
        
        student_pred, student_h = self.core(noisy, ctx, t, d_fast, output_hidden_states = True)
        sc_kd_loss = self.kd_loss(student_h, teacher_h)
        sc_loss = F.mse_loss(teacher_pred, student_pred)

        return sc_loss, sc_kd_loss
            
    def forward(self, x):
        # Split batch into regular and shortcut samples based on sc_batch_frac
        batch_size = len(x[0])

        if self.config.sc_weight > 0:
            n_regular = int(batch_size * (1 - self.config.sc_batch_frac))
            x, sc_x = x[:n_regular], x[n_regular:]
            x, ctx = x
            x, sc_x = x[:n_regular], x[n_regular:]
            ctx, sc_ctx = ctx[:n_regular], ctx[n_regular:]
            sc_x = (sc_x, sc_ctx)
            x = (x, ctx)
        
        x, ctx = x
        ctx = ctx.unsqueeze(1).repeat(1, 77, 1)
        ctx = ctx.to(x.dtype)
        neg_ctx = self.empty_embed.repeat(x.shape[0],1,1).to(x.dtype).to(x.device)
        
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

            teacher_pred, teacher_h = self.teacher(lerpd, ctx, t, d, output_hidden_states = True)
        
        extra = {}
        total_loss = 0.

        pred, student_h = self.core(lerpd, ctx, t, d, output_hidden_states = True)
        diff_loss = F.mse_loss(target, pred)
        kd_loss = self.kd_loss(student_h, teacher_h)

        total_loss += diff_loss
        total_loss += self.config.kd_weight * kd_loss * 0.5

        extra['kd_loss'] = kd_loss.item()

        if self.config.sc_weight > 0:
            sc_loss, kd_2 = self.get_sc_loss(sc_x)
            total_loss += (self.config.sc_weight * sc_loss) + (0.5 * self.config.kd_weight * kd_2)
            extra['sc_loss'] = sc_loss.item()
            extra['kd_loss'] = (extra['kd_loss'] * 0.5 + kd_2 * 0.5).item()
        else:
            extra['sc_loss'] = 0.
        
        extra['diff_loss'] = diff_loss.item()
        return total_loss, extra

        


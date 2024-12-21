from common.configs import SamplerConfig
from diffusers import FlowMatchEulerDiscreteScheduler
import torch

class CFGSampler:
    def __init__(self, config : SamplerConfig):
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3)
        self.config = config
    
    @torch.no_grad()
    def sample(self, n_samples = None, model = None, prompts = None):
        if n_samples is None:
            n_samples = len(prompts)

        n_steps = self.config.n_steps
        guidance_scale = self.config.cfg_scale

        assert prompts is not None, "Prompts cannot be None for CFGSampler"
        assert len(prompts) == n_samples, "Number of prompts must match number of samples"

        # 1. Double the prompts, adding empty strings
        prompts = prompts + [""] * len(prompts)
        c = model.encode_text(prompts)
            
        # 2. Make the noise
        sample_shape = (model.config.channels, model.config.sample_size, model.config.sample_size)
        sample_shape = (n_samples,) + sample_shape
        noisy = torch.randn(*sample_shape)

        # 3. Scheduler, timesteps and sigmas
        self.scheduler.set_timesteps(n_steps)
        timesteps = self.scheduler.timesteps / 1000
        sigmas = self.scheduler.sigmas

        # 3. Move everything to same device/dtype as model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        noisy = noisy.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device=device, dtype=dtype)
        sigmas = sigmas.to(device=device, dtype=dtype)
        c = c.to(device=device, dtype=dtype)

        for i, t in tqdm(enumerate(timesteps)):
            dt = sigmas[i+1] - sigmas[i]

            noisy_doubled = torch.cat([noisy, noisy], dim=0)            
            pred = model.denoise(noisy_doubled, c, t)
            pred_cond, pred_uncond = pred.chunk(2)
            v = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            noisy += v * dt

        if model.vae is None:
            return noisy
        else:
            return model.vae.decode(noisy)
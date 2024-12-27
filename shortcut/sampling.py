from common.configs import SamplerConfig
from diffusers import FlowMatchEulerDiscreteScheduler
import torch
from tqdm import tqdm

class SlowSampler:
    def __init__(self, config : SamplerConfig = SamplerConfig()):
        self.config = config
    
    @torch.no_grad()
    def sample(self, n_samples = None, model = None, prompts = None):
        if n_samples is None:
            n_samples = len(prompts)

        guidance_scale = self.config.cfg_scale

        assert prompts is not None, "Prompts cannot be None for SlowSampler"
        assert len(prompts) == n_samples, "Number of prompts must match number of samples"

        # 1. Double the prompts, adding empty strings
        prompts = prompts + [""] * len(prompts)
        c = model.encode_text(prompts)
            
        # 2. Make the noise
        sample_shape = (model.config.channels, model.config.sample_size, model.config.sample_size)
        sample_shape = (n_samples,) + sample_shape
        noisy = torch.randn(*sample_shape)

        # 3. Generate timesteps
        timesteps = torch.linspace(1, 0, 129)[:-1]  # 128 steps, excluding 0
        dt = -1/128

        # 4. Move everything to same device/dtype as model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        noisy = noisy.to(device=device, dtype=dtype)
        timesteps = timesteps.to(device=device, dtype=dtype)
        c = c.to(device=device, dtype=dtype)

        for t in tqdm(timesteps):
            noisy_doubled = torch.cat([noisy, noisy], dim=0)            
            pred = model.denoise(noisy_doubled, c, t, torch.full_like(t, 128))
            pred_cond, pred_uncond = pred.chunk(2)
            v = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            noisy += v * dt

        if model.vae is None:
            return noisy
        else:
            return model.vae.decode(noisy)

class FastSampler:
    def __init__(self, config : SamplerConfig = SamplerConfig()):
        self.config = config
    
    @torch.no_grad()
    def sample(self, n_samples = None, model = None, prompts = None):
        if n_samples is None:
            n_samples = len(prompts)

        assert prompts is not None, "Prompts cannot be None for OneStepSampler"
        assert len(prompts) == n_samples, "Number of prompts must match number of samples"

        # 1. Get text embeddings
        c = model.encode_text(prompts)
            
        # 2. Make the noise
        sample_shape = (model.config.channels, model.config.sample_size, model.config.sample_size)
        sample_shape = (n_samples,) + sample_shape
        noisy = torch.randn(*sample_shape)

        # 3. Move everything to same device/dtype as model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        noisy = noisy.to(device=device, dtype=dtype)
        c = c.to(device=device, dtype=dtype)
        t = torch.ones(n_samples, device=device, dtype=dtype)

        # 4. Single step denoising
        pred = model.denoise(noisy, c, t, torch.ones_like(t))
        noisy += pred * -1.0  # dt = -1 since we go from t=1 to t=0

        if model.vae is None:
            return noisy
        else:
            return model.vae.decode(noisy)
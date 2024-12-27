from common.configs import SamplerConfig
import torch
from tqdm import tqdm

class TinySampler:
    def __init__(self, config : SamplerConfig = SamplerConfig()):
        self.config = config
    
    @torch.no_grad()
    def sample(self, n_samples = None, model = None, prompts = None):
        if n_samples is None:
            n_samples = len(prompts)

        assert prompts is not None, "Prompts cannot be None for TinySampler"
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

        # 4. Single step denoising
        pred = model.denoise(noisy, c)

        if model.vae is None:
            return noisy + pred
        else:
            return model.vae.decode(noisy + pred)

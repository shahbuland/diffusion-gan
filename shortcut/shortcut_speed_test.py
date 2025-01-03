import torch
from common.configs import ModelConfig, SamplerConfig
from shortcut.model import RFTCore
from common.nn.vae import VAE
from common.nn.text_embedder import TextEmbedder
from shortcut.sampling import FastSampler, SlowSampler
import time
from tqdm import tqdm

from common.nn.mlp import MixFFNOpt
from common.nn.transformer import InferenceAttn
from common.nn.normalization import RMSNormOpt
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(True)

class ShortcutSampler:
    def __init__(self):
        # Load VAE and text encoder
        self.vae = VAE()
        self.vae.cuda()
        self.vae.eval()
        self.vae = self.vae.half() # Convert to fp16
        
        self.text_embedder = TextEmbedder() 
        self.text_embedder.cuda()
        self.text_embedder.eval()
        self.text_embedder = self.text_embedder.half()

        # Load model config and create model
        model_cfg = ModelConfig.from_yaml("configs/dit_large.yml")
        self.model = RFTCore(model_cfg)

        # Load checkpoint (filtering for core params only)
        state_dict = torch.load("checkpoints/dit_l_275k/ema_model.pth")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('ema_model.core.'):
                new_key = key.replace('ema_model.core.', '')
                new_state_dict[new_key] = value
        
        self.model.load_state_dict(new_state_dict)
        self.model.cuda()
        self.model.half()
        self.model.eval()

        # Replace certain training modules with optimized versions
        for i in range(24):
            # Replace MLP
            old_mlp = self.model.blocks.blocks[i].mlp
            new_mlp = MixFFNOpt(model_cfg)
            new_mlp.cuda()
            new_mlp.half()
            
            # Copy weights from old MLP to new optimized version
            with torch.no_grad():
                # Copy conv1 weights to w1, b1
                new_mlp.w1.data = old_mlp.conv_1.weight.view(old_mlp.conv_1.weight.size(0), -1)
                new_mlp.b1.data = old_mlp.conv_1.bias.data
                
                # Copy conv2 weights (depthwise conv stays the same)
                new_mlp.conv_2.weight.data = old_mlp.conv_2.weight.data
                new_mlp.conv_2.bias.data = old_mlp.conv_2.bias.data
                
                # Copy conv3 weights to w3, b3
                new_mlp.w3.data = old_mlp.conv_3.weight.view(old_mlp.conv_3.weight.size(0), -1)
                new_mlp.b3.data = old_mlp.conv_3.bias.data
            
            # Replace the MLP
            self.model.blocks.blocks[i].mlp = new_mlp

            # Replace Attention
            old_attn = self.model.blocks.blocks[i].attn
            new_attn = InferenceAttn(model_cfg)
            new_attn.cuda()
            new_attn.half()

            # Copy weights from old attention to new attention
            with torch.no_grad():
                new_attn.qkv.weight.data = old_attn.qkv.weight.data
                new_attn.out.weight.data = old_attn.out.weight.data
                
                # Get gain values from old RMSNorm modules
                q_gain = old_attn.q_norm.g.data
                k_gain = old_attn.k_norm.g.data
                
                # Create new optimized RMSNorm with the gain values
                new_attn.q_norm = RMSNormOpt(q_gain).cuda().half()
                new_attn.k_norm = RMSNormOpt(k_gain).cuda().half()

            # Replace the attention module
            self.model.blocks.blocks[i].attn = new_attn
    
        self.text_embedder.encode_text = torch.compile(self.text_embedder.encode_text)
        self.vae.decode = torch.compile(self.vae.decode)
        self.model = torch.compile(self.model)
    
        self.cached_prompt = None

    def set_prompt(self, prompt):
        self.cached_prompt = self.text_embedder.encode_text([prompt]).half()

    @torch.no_grad()
    def sample(self, prompt = None):
        n_samples = 1

        # Get text embeddings
        if prompt is None:
            c = self.cached_prompt
        else:
            c = self.text_embedder.encode_text([prompt]).half()
        
        # Make the noise 
        sample_shape = (64, 8, 8)
        sample_shape = (n_samples,) + sample_shape
        noisy = torch.randn(*sample_shape)

        # Move everything to GPU and convert to fp16
        device = 'cuda'
        noisy = noisy.to(device=device, dtype=torch.float16)
        c = c.to(device=device)

        # Generate 4 evenly spaced timesteps from 1 to 0
        n_steps = 4
        timesteps = torch.linspace(1, 0, n_steps+1)[:-1]  # 4 steps, excluding 0
        dt = -1/n_steps

        d = torch.full((n_samples,), n_steps, device=device, dtype=torch.float16)

        for t in timesteps:
            t = t.to(device=device, dtype=torch.float16)
            pred = self.model(noisy, c, t, d)
            noisy += pred * dt

        # Decode with VAE
        return self.vae.decode(noisy)


if __name__ == "__main__":
    sampler = ShortcutSampler()

    # Set prompt
    sampler.set_prompt("A photo of a dog")

    # Warmup call
    print("Doing warmup call...")
    with torch.no_grad():
        samples = sampler.sample()

    # Benchmark fast sampling
    print("\nBenchmarking 20 calls...")
    times = []
    for i in range(20):
        start = time.time()
        with torch.no_grad():
            samples = sampler.sample()
        times.append(time.time() - start)
        fps = 1.0 / times[-1]
        print(f"Call {i+1}/20: {fps:.2f} FPS")

    print(f"\nResults:")
    print(f"Max FPS: {1.0/min(times):.2f}")
    print(f"Avg FPS: {1.0/(sum(times)/len(times)):.2f}")
    print(f"Min FPS: {1.0/max(times):.2f}")

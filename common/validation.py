import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize, ToTensor
import os
from tqdm import tqdm
import joblib

class FIDScorer:
    def __init__(self, cache_path='./fid_cache.pkl', total_size=10000, batch_size=256, device='cuda', n_sampling_steps : int = 32):
        from common.data import create_loader
        self.loader = create_loader('coco', batch_size=batch_size, image_size=512, split='validation')
        self.total_size = total_size
        self.batch_size = batch_size
        self.device = device
        self.cache_path = cache_path
        self.fid = FrechetInceptionDistance(feature=2048).to(device)

        if os.path.exists(self.cache_path):
            self.load(self.cache_path)
        else:
            self._compute_real_stats()

    def load(self, path):
        real_stats = joblib.load(path)
        real_f_sum, real_f_cov_sum, real_f_n = real_stats
        self.fid.real_features_sum = real_f_sum
        self.fid.real_features_cov_sum = real_f_cov_sum
        self.fid.real_features_num_samples = real_f_n
        print(f"Loaded FID statistics for {real_f_n} real images from {path}")

    def save(self, path):
        real_f_sum = self.fid.real_features_sum
        real_f_cov_sum = self.fid.real_features_cov_sum
        real_f_n = self.fid.real_features_num_samples
        joblib.dump([real_f_sum, real_f_cov_sum, real_f_n], path)

    @torch.no_grad()
    def _compute_real_stats(self):
        print("Computing FID statistics for real images...")
        processed_samples = 0
        all_images = []
        self.fid.reset()
        for images, _ in tqdm(self.loader):
            if processed_samples >= self.total_size:
                break
            images = (images * 255).byte().to(self.device)
            self.fid.update(images, real = True)
            processed_samples += images.shape[0]
        
        print(f"Processed {processed_samples} real images for FID calculation.")
        self.save(self.cache_path)

    @torch.no_grad()
    def __call__(self, sampler, model):
        self.fid.reset()
        self.load(self.cache_path)

        print("Generating images for FID calculation...")
        prompts = []
        for _, batch_prompts in self.loader:
            prompts.extend(batch_prompts)
            if len(prompts) >= self.total_size:
                prompts = prompts[:self.total_size]
                break

        print(f"Total prompts: {len(prompts)}")

        all_images = []

        for i in tqdm(range(0, self.total_size, self.batch_size)):
            batch_size = min(self.batch_size, self.total_size - i)
            batch_prompts = prompts[i:i+batch_size]
            
            images = sampler.sample(batch_size, model, batch_prompts)
            images = (images * 255).byte().to(self.device)
            self.fid.update(images, real = False)
           
        return self.fid.compute().item()
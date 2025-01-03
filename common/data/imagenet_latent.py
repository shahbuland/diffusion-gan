import os
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm
import shutil
from pathlib import Path
import numpy as np
from tinygrad.helpers import Timing

from common.nn.vae import VAE
from common.nn.text_embedder import TextEmbedder

CACHE_DIR = "latent_dataset_cache"

class LatentImageNetDataset(Dataset):
    def __init__(self, split='train', device = 'cpu'):
        self.split = split
        self.cache_dir = Path(CACHE_DIR) / split
        
        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self._create_cache()
        else:
            # Validate cache
            self._validate_cache()
            
        # Load cached data
        self.latents = torch.load(self.cache_dir / "image_latents.pt", map_location=device)
        self.text_embeds = torch.load(self.cache_dir / "text_embeds.pt", map_location=device)
    
    def shuffle(self):
        """Randomly shuffles the dataset by permuting the latents and text embeddings together"""
        perm = torch.randperm(len(self.latents))
        self.latents = self.latents[perm]
        self.text_embeds = self.text_embeds[perm]
        
    @torch.no_grad()
    def _create_cache(self):
        print(f"Creating cache for {self.split} split...")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        vae = VAE().cuda().to(torch.bfloat16)
        vae.encoder = torch.compile(vae.encoder)
        text_embedder = TextEmbedder().cuda().to(torch.bfloat16)
        text_embedder = torch.compile(text_embedder)

        # Load dataset
        dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split=self.split)
        
        # Create dataloader for efficient batch processing
        from torch.utils.data import DataLoader
        
        class ProcessingDataset(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset
                self.label_names = dataset.features['label'].names
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                item = self.dataset[idx]
                image = torch.from_numpy(np.array(item['image'])).permute(2,0,1).float() / 127.5 - 1
                label = self.label_names[item['label']]
                return image, label
        
        processing_dataset = ProcessingDataset(dataset)
        loader = DataLoader(
            processing_dataset,
            batch_size=4096,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            shuffle=False
        )
        
        all_latents = []
        all_text_embeds = []

        for images, labels in tqdm(loader):
            images = images.to('cuda')
            
            with torch.no_grad():
                latents = vae.encode(images)
                all_latents.append(latents.cpu())
                
                text_embeds = text_embedder.encode_text(labels).mean(1)  # Pool embeddings
                all_text_embeds.append(text_embeds.cpu())
        # Concatenate and save
        all_latents = torch.cat(all_latents)
        all_text_embeds = torch.cat(all_text_embeds)
        
        torch.save(all_latents, self.cache_dir / "image_latents.pt")
        torch.save(all_text_embeds, self.cache_dir / "text_embeds.pt")
        
        print(f"Cache created at {self.cache_dir}")
        
    def _validate_cache(self):
        if not (self.cache_dir / "image_latents.pt").exists() or \
           not (self.cache_dir / "text_embeds.pt").exists():
            print("Cache validation failed - recreating cache")
            shutil.rmtree(self.cache_dir)
            self._create_cache()
            
    def __len__(self):
        return len(self.latents)
        
    def __getitem__(self, idx):
        return self.latents[idx], self.text_embeds[idx]

# Custom dataloader optimized for pre-cached GPU tensors
class LatentDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Pre-calculate number of batches
        self.n_samples = len(dataset)
        self.n_batches = self.n_samples // batch_size
        if not drop_last and self.n_samples % batch_size != 0:
            self.n_batches += 1
            
        # Create index buffer
        self.indices = torch.arange(self.n_samples)
        
    def __iter__(self):
        if self.shuffle:
            # In-place shuffle of indices
            rand_idx = torch.randperm(self.n_samples)
            self.indices = self.indices[rand_idx]
        
        for i in range(self.n_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            
            # Handle last batch if not dropping
            if end_idx > self.n_samples:
                if self.drop_last:
                    break
                end_idx = self.n_samples
            
            # Get batch indices
            batch_indices = self.indices[start_idx:end_idx]
            
            # Efficiently index both tensors at once
            latents = self.dataset.latents[batch_indices]
            embeds = self.dataset.text_embeds[batch_indices]
            
            yield latents.to('cuda'), embeds.to('cuda')
            
    def __len__(self):
        return self.n_batches

# Test the dataloader
if __name__ == "__main__":
    import time
    
    dataset = LatentImageNetDataset()
    loader = LatentDataLoader(dataset, batch_size=256)
    
    print(f"Number of batches: {len(loader)}")
    
    # Test iteration and timing
    total_time = 0
    start = time.perf_counter()
    for i, (latents, embeds) in enumerate(loader):

        print(latents.device)
        print(embeds.device)
        print(latents.dtype)
        exit()
        
        
        if i == 0:
            print(f"First batch shapes:")
            print(f"Latents: {latents.shape}")
            print(f"Embeddings: {embeds.shape}")
            
        batch_time = time.perf_counter() - start
        total_time += batch_time
        start = time.perf_counter()
        
        print(f"Batch {i}: {batch_time*1000:.2f}ms")
        
    print(f"\nAverage batch load time: {(total_time/len(loader))*1000:.2f}ms")

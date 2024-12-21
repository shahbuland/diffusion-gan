import math
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import nn
import einops as eo 
import math

# Shortcut model specific things

def log2(x):
    return math.log(x, 2)

def sample_step_size(batch_size, max_steps = 128):
    """
    Excluding 2^0, samples possible step sizes, i.e. 2, 4, ... 128
    Return type is [b,] tensor
    """
    # Calculate the number of possible step sizes (excluding 2^0)
    num_options = int(log2(max_steps))
    
    # Generate possible step sizes: 2^1, 2^2, ..., 2^log2(max_steps)
    possible_steps = torch.tensor([2**i for i in range(1, num_options + 1)])
    
    # Sample from possible steps with replacement
    sampled_steps = torch.randint(0, num_options, (batch_size,))
    
    # Convert sampled indices to actual step sizes
    return possible_steps[sampled_steps]

def sample_discrete_timesteps(n_steps):
    """
    Sample timestamps that make sense given n_steps
    """
    # n_steps is a [b,] tensor of values like [1, 2, 4, 8, 16, ...]

    # This code is weird so I will explain:
    # For each n_steps value n:
    # - generate possible values as a range from 0 to 1 in n_steps, excluding 1
    # - i.e. if n = 2, linspace(0,1) -> [0,1] which is wrong, we want [0, 0.5]
    # - then select random value from this range
    # - t = 0 is useless, it means image isn't noised, so do 1 - t
    
    # Generate possible timesteps for each batch element
    def _round(n):
        return round(n.item())
    t = [torch.linspace(0,1,steps=_round(n)+1)[:-1][torch.randint(0, _round(n), (1,))] for n in n_steps]
    return 1 - torch.tensor(t, device = n_steps.device, dtype = n_steps.dtype)

# ===============================

def count_parameters(model):
    """
    Count and print the number of learnable parameters in a model.
    
    Args:
        model (nn.Module): The PyTorch model to analyze.
    """
    total_params = sum(p.numel() for p in model.core.parameters() if p.requires_grad)
    return total_params

def pretty_print_parameters(model):
    """
    Same as above func but doesn't return anything, just prints in a pretty format
    """
    params = count_parameters(model)
    formatted_params = params
    if params < 1_000_000:
        formatted_params = f"{params // 1000}K"
    elif params < 1_000_000_000:
        formatted_params = f"{params // 1_000_000}M"
    elif params < 1_000_000_000_000:
        formatted_params = f"{params // 1_000_000_000}B"
    else:
        formatted_params = f"{params // 1_000_000_000_000}T"
    
    print(f"Model has {formatted_params} trainable parameters.")

def freeze(module: nn.Module):
    """
    Set all parameters in a module to not require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to freeze.
    """
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module: nn.Module):
    """
    Set all parameters in a module to require gradients.
    
    Args:
        module (nn.Module): The PyTorch module to unfreeze.
    """
    for param in module.parameters():
        param.requires_grad = True

import time

class Stopwatch:
    def __init__(self):
        self.start_time = None

    def reset(self):
        """Prime the stopwatch for measurement."""
        self.start_time = time.time()

    def hit(self, samples: int) -> float:
        """
        Measure the average time per 1000 samples since the last reset.

        Args:
            samples (int): The number of samples processed.

        Returns:
            float: The time in seconds per 1000 samples.
        """
        if self.start_time is None:
            raise ValueError("Stopwatch must be reset before calling hit.")

        elapsed_time = time.time() - self.start_time
        avg_time_per_sample = elapsed_time / samples
        return avg_time_per_sample * 1000  # Return time per 1000 samples

import torch
import wandb
import numpy as np

def to_wandb_image(x : TensorType["c", "h", "w"], caption : str = ""):
    """
    Turn tensor into wandb image for sampling
    """
    x = eo.rearrange(x, 'c h w -> h w c')
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    x = x.detach().cpu().numpy()
    return wandb.Image(x, caption = caption)

def to_wandb_batch(x, captions = None):
    if captions is None:
        return [to_wandb_image(x_i) for x_i in x]
    else:
        return [to_wandb_image(x_i, caption) for (x_i, caption) in zip(x, captions)]
from torch.optim.lr_scheduler import _LRScheduler
import math

def get_scheduler_cls(scheduler_name: str):
    """
    Returns the scheduler class based on the given name.

    Args:
        scheduler_name (str): The name of the scheduler.

    Returns:
        _LRScheduler: The scheduler class.

    Raises:
        ValueError: If an invalid scheduler name is provided.
    """
    scheduler_map = {
        "CosineDecayAfterWarmup": CosineDecayAfterWarmup,
        "CosineDecay": CosineDecay,
        "LinearWarmup" : LinearWarmup,
        "Staircase" : StaircaseScheduler
    }

    scheduler_cls = scheduler_map.get(scheduler_name)
    if scheduler_cls is None:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}")
    
    return scheduler_cls

class LinearWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Keep constant after warmup
            return self.base_lrs

class CosineDecayAfterWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecayAfterWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) for base_lr in self.base_lrs]
        elif self.last_epoch < self.warmup_steps + self.T_max:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_steps) / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]

class CosineDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineDecay, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_max:
            # Cosine decay
            progress = self.last_epoch / self.T_max
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        else:
            # Constant minimum learning rate
            return [self.eta_min for _ in self.base_lrs]

class StaircaseScheduler(_LRScheduler):
    def __init__(self, optimizer, ramp_down_steps, breakpoints, lr_values, last_epoch=-1):
        """
        Custom scheduler with fixed LR periods and linear ramp downs.
        
        Args:
            optimizer: PyTorch optimizer
            ramp_down_steps (int): Number of steps for each ramp down period
            breakpoints (list): List of steps where ramp downs begin
            lr_values (list): Target LR values after each ramp down
            last_epoch (int): The index of last epoch
        """
        self.ramp_down_steps = ramp_down_steps
        self.breakpoints = breakpoints
        self.lr_values = lr_values
        
        # Validate inputs
        if len(breakpoints) != len(lr_values):
            raise ValueError("breakpoints and lr_values must have the same length")
        if not all(x < y for x, y in zip(breakpoints[:-1], breakpoints[1:])):
            raise ValueError("breakpoints must be strictly increasing")
            
        super(StaircaseScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get initial learning rate from optimizer
        initial_lr = self.base_lrs[0]
        
        # If we haven't reached the first breakpoint, return initial learning rate
        if self.last_epoch < self.breakpoints[0]:
            return [initial_lr for _ in self.base_lrs]
        
        # Find the current segment
        current_segment = 0
        for i, bp in enumerate(self.breakpoints):
            if self.last_epoch >= bp:
                current_segment = i
            else:
                break
                
        # Get start and end LR for current segment
        start_lr = initial_lr if current_segment == 0 else self.lr_values[current_segment - 1]
        target_lr = self.lr_values[current_segment]
        
        # If we're in a ramp down period
        if self.last_epoch < self.breakpoints[current_segment] + self.ramp_down_steps:
            # Calculate progress through ramp down
            progress = (self.last_epoch - self.breakpoints[current_segment]) / self.ramp_down_steps
            current_lr = start_lr + progress * (target_lr - start_lr)
        else:
            # We're in a steady state period
            current_lr = target_lr
            
        return [current_lr for _ in self.base_lrs]

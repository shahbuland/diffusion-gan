from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import yaml

@dataclass
class ConfigClass:
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'ConfigClass':
        with open(yaml_file, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

    def save_yaml(self, yaml_file: str) -> None:
        with open(yaml_file, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

@dataclass
class ModelConfig(ConfigClass):
    # Transformer
    n_layers : int = 28
    n_heads : int = 16
    d_model : int = 1152
    flash : bool = True
    normalized : bool = False

    # input/latent
    image_size : int = 256
    sample_size : int = 8
    channels : int = 64
    patch_size : int = 1
    use_vae = True
    
    # Guidance
    take_label : bool = True # Take the batch as (pixel_values, label_str) instead of pixel_values
    text_d_model : int = 512 # Hidden size of text embedding model
    cfg_prob : float = 0.1

    # Shortcut models
    sc_weight : float = 1.0
    sc_cfg : float = 3.5
    sc_batch_frac : float = 0.25 # What percentage of training batch?
    delay_sc : int = 0 # Delay sc to this many steps after training starts
    base_steps : int = 128

    # Knowledge distillation
    kd_weight : float = 1.0
    kd_inds_student : List[int] = field(default_factory = lambda : [
        2, 8, 12, 15
    ])
    kd_inds_teacher : List[int] = field(default_factory = lambda : [
        4, 12, 24, 27
    ])
    # defaults assume 28 layer teacher, 16 layer student

@dataclass
class TrainConfig(ConfigClass):
    dataset : str = "coco"
    target_batch_size : int = 256
    batch_size : int = 256
    epochs : int = 100
    # optimizer
    opt : str = "Muon"
    opt_kwargs : Dict = field(default_factory = lambda : {
        "lr": 1.0e-4,
        "eps": 1.0e-15,
        "betas" : (0.9, 0.95),
        #'lr_1d' : 1.0e-3,
        #"b1" : 0.9,
        "weight_decay" : 0.1,
        #"precondition_frequency" : 2
    })

    scheduler : Optional[str] = None
    scheduler_kwargs : Dict = field(default_factory = lambda: {
        "ramp_down_steps": 1000,
        "breakpoints": [20000, 40000, 80000],
        "lr_values": [5.0e-4, 1.0e-4, 5.0e-5]
    })

    # Saving
    checkpoint_root_dir = "checkpoints"

    log_interval : int = 1
    sample_interval : int = 500
    save_interval : int = 20000
    val_interval : int = 5000
    resume : bool = False
    resume_path : str = "checkpoints/40k_resume"

    # Sampling
    n_samples : int = 16 # Number of samples to log each time (too many gets crowded)
    sample_prompts : List[str] = field(default_factory = lambda: [
        "golden retriever",
        "tabby cat",
        "school bus",
        "tennis ball",
        "acoustic guitar",
        "monarch butterfly",
        "great white shark",
        "bald eagle",
        "red panda",
        "pineapple",
        "fire truck",
        "grand piano",
        "mountain bike",
        "polar bear",
        "peacock",
        "zebra"
    ])
    
    # Validating
    val_batch_mult = 2

    # Adversarial training details
    adv_weight : float = 0.75
    delay_adv : int = 15000
    adv_warmup : int = 5000

@dataclass
class LoggingConfig:
    run_name : str = "baseline dit base"
    wandb_entity : str = "shahbuland"
    wandb_project : str = "diffusion-gan"

@dataclass
class SamplerConfig:
    n_steps : int = 128
    cfg_scale : float = 1.5
    fast_steps : int = 1
from PIL import Image

import torch
from common .data import create_loader
from .model import TeacherStudentRFT
from .trainer import Trainer
from common.configs import ModelConfig, TrainConfig, LoggingConfig
from common.utils import pretty_print_parameters


"""
Some notes:

Tiny DiT with same width = 214M params
DiT Large = 400M params
"""
if __name__ == "__main__":
    #model_cfg = ModelConfig.from_yaml("configs/tiny_dit_maintain_width.yml")
    #model_cfg = ModelConfig.from_yaml("configs/dit_large.yml")
    model_cfg = ModelConfig.from_yaml("configs/tiny_dit.yml")
    teacher_cfg = ModelConfig.from_yaml("configs/dit_large.yml")
    teacher_ckpt_path = "checkpoints/dit_l_best_fid/ema_model.pth"

    train_cfg = TrainConfig.from_yaml("configs/adamw.yml")
    log_cfg = LoggingConfig()
    
    seed = 42
    torch.manual_seed(seed)

    model = TeacherStudentRFT(model_cfg, teacher_cfg, teacher_ckpt_path)
    trainer = Trainer(train_cfg, log_cfg, model_cfg)
    
    # Create the data loader using the configuration
    train_loader = create_loader(
        dataset_name=train_cfg.dataset,
        batch_size=train_cfg.batch_size,
        image_size=model_cfg.image_size,
        deterministic=True  # This will use a fixed seed internally
    )

    trainer.train(model, train_loader)
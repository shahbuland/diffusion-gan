import torch
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import os
from dataclasses import asdict
from ema_pytorch import EMA

from common.configs import TrainConfig, LoggingConfig, ModelConfig
from common.optimizers import get_extra_optimizer
from common.schedulers import get_scheduler_cls
from common.utils import Stopwatch
from .sampling import TinySampler
from common.wandb_utils import *
from common.validation import FIDScorer

class Trainer:
    def __init__(self, config : TrainConfig, logging_config : LoggingConfig = None, model_config : ModelConfig = None):
        self.config = config
        self.logging_config = logging_config
        self.model_config = model_config

        self.accum_steps = self.config.target_batch_size // self.config.batch_size
        self.accelerator = Accelerator(
            log_with = "wandb",
            gradient_accumulation_steps = self.accum_steps
        )

        tracker_kwargs = {}
        if self.logging_config is not None:
            log = self.logging_config
            tracker_kwargs['wandb'] = {
                'name' : log.run_name,
                'entity' : log.wandb_entity,
                'mode' : 'online'
            }

            config_dict = asdict(config)
            if model_config is not None:
                config_dict.update(asdict(model_config))


            self.accelerator.init_trackers(
                project_name = log.wandb_project,
                config = config_dict,
                init_kwargs = tracker_kwargs
            )

        self.world_size = self.accelerator.state.num_processes
        self.total_step_counter = 0
        self.ema = None

    def get_should(self, step = None):
        # Get a dict of bools that determines if certain things should be done at the current step
        if step is None:
            step = self.total_step_counter

        def should_fn(interval):
            return step % interval == 0 and self.accelerator.sync_gradients

        return {
            "log" : should_fn(self.config.log_interval),
            "save" : should_fn(self.config.save_interval),
            "sample" : should_fn(self.config.sample_interval),
            "val" : should_fn(self.config.val_interval)
        }

    def save(self, step = None, dir = None):
        """
        In directory, save checkpoint of accelerator state using step and self.logging_config.run_name
        """
        if step is None:
            step = self.total_step_counter
        if dir is None:
            dir = os.path.join(self.config.checkpoint_root_dir, f"{self.logging_config.run_name}_{step}")
        
        os.makedirs(dir, exist_ok = True)

        self.accelerator.save_state(output_dir = dir)
        if self.ema is not None:
            ema_path = os.path.join(dir, "ema_model.pth")
            torch.save(self.ema.state_dict(), ema_path)
            ema_model_path = os.path.join(dir, "out.pth")
            torch.save(self.ema.ema_model.state_dict(), ema_model_path)
    
    def load(self):
        """
        Load the latest checkpoint from the checkpoint directory.
        """
        checkpoint_dir = self.config.checkpoint_root_dir
        run_name = self.logging_config.run_name
        
        # Get all directories that match the run name pattern
        matching_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith(run_name) and d[len(run_name):].strip('_').isdigit()]
        
        if not matching_dirs:
            print(f"No checkpoints found for {run_name}")
            _ = input("Are you sure you want to continue?")
            return
        
        # Sort directories by the number at the end and get the latest
        latest_dir = max(matching_dirs, key=lambda x: int(x[len(run_name):].strip('_')))
        latest_checkpoint = os.path.join(checkpoint_dir, latest_dir)
        
        print(f"Loading checkpoint from {latest_checkpoint}")
        
        # Load accelerator state
        self.accelerator.load_state(latest_checkpoint)
        
        # Load EMA if it exists
        ema_path = os.path.join(latest_checkpoint, "ema_model.pth")
        if os.path.exists(ema_path) and self.ema is not None:
            self.ema.load_state_dict(torch.load(ema_path))
        
        print("Checkpoint loaded successfully")

    def train(self, model, discriminator, loader, val_loader = None):
            # optimizer setup 
            try:
                opt_class = getattr(torch.optim, self.config.opt)
            except:
                opt_class = get_extra_optimizer(self.config.opt)

            if self.config.opt.lower() == "muon":
                opt = opt_class(
                    model.parameters('main'),
                    lr=self.config.opt_kwargs['lr'],
                    adamw_params=model.parameters('projection'),
                    adamw_lr=self.config.opt_kwargs['adam_lr'],
                    adamw_betas=self.config.opt_kwargs['betas'],
                    adamw_eps=self.config.opt_kwargs['eps'],
                    adamw_wd=self.config.opt_kwargs['weight_decay']
                )
            else:
                opt = opt_class(model.parameters(), **self.config.opt_kwargs)

            d_opt = torch.optim.AdamW(
                discriminator.parameters(),
                lr=3.0e-5,
                eps=1.0e-15,
                betas=(0.9, 0.95),
                weight_decay=0.01
            )

            # scheduler setup
            scheduler = None
            if self.config.scheduler is not None:
                try:
                    scheduler_class = getattr(torch.optim.lr_scheduler, self.config.scheduler)
                except:
                    scheduler_class = get_scheduler_cls(self.config.scheduler)
                scheduler = scheduler_class(opt, **self.config.scheduler_kwargs)

            # accelerator prepare
            model.train()
            discriminator.train()
            if scheduler:
                model, loader, opt, scheduler = self.accelerator.prepare(model, loader, opt, scheduler)
            else:
                model, loader, opt = self.accelerator.prepare(model, loader, opt)

            discriminator, d_opt = self.accelerator.prepare(discriminator, d_opt)

            # ema setup
            self.ema = EMA(
                self.accelerator.unwrap_model(model),
                beta = 0.9999,
                update_after_step = 1,
                update_every = 1,
                ignore_names = {'vae', 'text_embedder', 'kd_loss', 'teacher'},
                coerce_dtype = True
            )
            accel_ema = self.accelerator.prepare(self.ema)

            # load checkpoint if we want to 
            if self.config.resume:
                self.accelerator.load_state(self.config.resume_path)
                
            sw = Stopwatch()
            sw.reset()

            if self.logging_config is not None:
                wandb.watch(self.accelerator.unwrap_model(model), log = 'all')

            # Set up samplers
            sampler = TinySampler()

            # Set up FID scorer
            fid_scorer = FIDScorer(
                batch_size=self.config.batch_size * self.config.val_batch_mult
            )

            def get_grad_norm(grads):
                return torch.sqrt(sum(torch.sum(g * g) for g in grads))
                
            def adaptive_backward(rec_loss, adv_loss, adv_weight):
                rec_grads = torch.autograd.grad(rec_loss, model.parameters(), retain_graph=True, allow_unused=True)
                adv_grads = torch.autograd.grad(adv_loss, model.parameters(), allow_unused=True)
                
                rec_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(rec_grads, model.parameters())]
                adv_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(adv_grads, model.parameters())]

                rec_norm = get_grad_norm(rec_grads)
                adv_norm = get_grad_norm(adv_grads)

                ada_w = (rec_norm / (adv_norm+1.0e-6)).detach()
                ada_w = self.accelerator.reduce(ada_w, reduction='mean')

                grads = [r + ada_w * adv_weight * a for r, a in zip(rec_grads, adv_grads)]
                
                for param, grad in zip(model.parameters(), grads):
                    if self.accelerator.scaler is not None:
                        grad = self.accelerator.scaler.scale(grad)
                    param.grad = (param.grad if param.grad is not None else 0) + grad.to(param.dtype) / self.accelerator.gradient_accumulation_steps

                return ada_w

            def warmup_factor(crnt_steps, total_steps):
                if total_steps == 0:
                    return 1
                return min(1, crnt_steps / total_steps)

            def warmup_adv_weight():
                base_weight = self.config.adv_weight
                total_paired_steps = self.total_step_counter - self.config.delay_adv

                if total_paired_steps <= 0:
                    return 0
                else:
                    return warmup_factor(total_paired_steps, self.config.adv_warmup) * base_weight

            ada_w = 0
            for epoch in range(self.config.epochs):
                for i, batch in enumerate(loader):
                    with self.accelerator.accumulate(model, discriminator):
                        opt.zero_grad()
                        d_opt.zero_grad()

                        total_loss, extra = model(batch)

                        if self.config.adv_weight > 0.0:
                            disc_loss = discriminator(extra['samples'].detach(), extra['original'].detach())
                            self.accelerator.backward(disc_loss)
                            d_opt.step()
                            
                        if warmup_adv_weight() > 0:
                            adv_loss = discriminator(extra['samples'])
                            extra['adv_loss'] = adv_loss.item()
                            ada_w = adaptive_backward(total_loss, adv_loss, warmup_adv_weight())
                            extra['adaptive_weight'] = ada_w.item()
                        else:
                            self.accelerator.backward(total_loss)
                        
                        opt.step()
                        if scheduler:
                            scheduler.step()

                        if self.accelerator.sync_gradients:
                            self.total_step_counter += 1
                            self.ema.update()

                        should = self.get_should()
                        if self.logging_config is not None and should['log'] or should['sample']:
                            wandb_dict = {
                                "loss": extra['diff_loss'],
                                "kd_loss": extra['kd_loss'],
                                "time_per_1k" : sw.hit(self.config.log_interval),
                                'disc_loss' : disc_loss.item()
                            }
                            if 'adv_loss' in extra:
                                wandb_dict['adv_loss'] = extra['adv_loss']
                                wandb_dict['adaptive_weight'] = extra['adaptive_weight']
                            if scheduler:
                                wandb_dict["learning_rate"] = scheduler.get_last_lr()[0]
                            if should['sample']:
                                n_samples = self.config.n_samples
                                images = to_wandb_batch(
                                    sampler.sample(n_samples, self.ema.ema_model, self.config.sample_prompts),
                                    self.config.sample_prompts
                                )
                                wandb_dict["samples"] = images
                            wandb.log(wandb_dict)
                            sw.reset()
                        if should['save']:
                            self.save(self.total_step_counter)
                        if should['val']:
                            self.ema.ema_model.eval()
                            # Calculate FID for both samplers
                            fid = fid_scorer(sampler, self.ema.ema_model)
                            self.ema.ema_model.train()
                            
                            wandb.log({
                                'fid': fid
                            })
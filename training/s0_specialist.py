# @GonzaloMartinGarcia
# Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.
# This training code is a modified version of the original text-to-image SD training code from the HuggingFace Inc. Team,
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py.
  
import logging
import math
import argparse
import os
import os.path as osp
import shutil
import random
from omegaconf import OmegaConf
import sys
import itertools
import cv2
from PIL import Image
import numpy as np
import imageio as iio
import accelerate
import datasets
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from dataloaders import UnifiedDataLoader
from util.noise import pyramid_noise_like
from util.lr_scheduler import IterExponential, IterLinear, IterConstant
from util.utils import overwrite_args, get_training_vis, get_latent_vis
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unid import UnidPipeline
from unid.util.rgb_projector import TaskProjector

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

class Trainer:
    """
    Trainer class for training the UNID model.
    This class handles the training loop, model saving, and evaluation.
    """
    
    def __init__(self, config_path, ow_args=None):
        """ Initialize the Trainer with the configuration file path.
        """
        
        self.args = self._load_config(config_path, ow_args)
        self._check_args_conditions()

        self.logger, self.accelerator = self._setup_logging_accelerator()
        if self.args.seed is not None:
            set_seed(self.args.seed, device_specific=True)
        
        self.train_dataloader = self._setup_train_loader()
        self._setup_pretrained_model()
        self.pipeline = self._setup_pipeline()
        self.unet = self.pipeline.unet
        self.projectors, self.proj_names = self._setup_projectors()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        
        # Prepare everything with `self.accelerator` (Move to GPU)
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        for proj_name, projector in self.projectors.items():
            self.projectors[proj_name] = self.accelerator.prepare(self.projectors[proj_name])
        self.device       = self.accelerator.device
        self.weight_dtype = self._get_weights_dtype()

        if self.accelerator.is_main_process:
            self._setup_tracker()

        self.task_to_tid = {task: i for i, task in enumerate(self.args.tasks.keys())}
        self.loss_functions, self.loss_item_id = self._setup_loss_functions()
        self.task_encodings, self.task_to_text_prompt = self._get_task_encodings()
    
    def train(self):
        """ Main training loop for the UNID model.
        """
        self.global_step = 0
        self._init_train_logging()

        if self.args.resume_from_checkpoint:
            self.resume_from_checkpoint()

        progress_bar = tqdm(
            range(0, self.args.max_train_steps), 
            initial=self.global_step, 
            desc="Steps", 
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process
        )
    
        # Training Loop
        train_loss = 0.0
        train_loss_per_item = {item: 0.0 for item in self.loss_item_id.keys()}
        train_cnt_per_item = {item: 1e-6 for item in self.loss_item_id.keys()}

        for step, batch in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.unet):
                dev_bs, T, _, _, _ = batch["rgb"].shape  # B, T, C, H, W
                
                #### === Prepare inputs === ####
                input_rgb_bt_flat = batch["rgb"].view(dev_bs*T, *batch["rgb"].shape[2:])  # B*T, C, H, W
                rgb_latents, noisy_latents = self.pipeline.prepare_input_latents(
                    input_rgb_bt_flat, self.args.noise_type, self.device, self.weight_dtype
                )
                rgb_latents = rgb_latents.view(dev_bs, T, *rgb_latents.shape[1:])           # B, T, 4, h, w
                noisy_latents = noisy_latents.view(dev_bs, T, *noisy_latents.shape[1:])     # B, T, 4, h, w
                anno_latents = batch.get("anno_latent", None)

                # Set timesteps to the first denoising step and task encoding for the batch
                timesteps = torch.ones((dev_bs,), device=self.device) * (self.pipeline.scheduler.config.num_train_timesteps-1) # 999
                timesteps = timesteps.long()
                encoder_hidden_states = self.task_encodings.repeat(dev_bs, 1, 1)

                #### === Forward pass === ####
                self.pipeline.init(is_video=batch["is_video"], is_train=True)
                current_latent_estimate_ls = list()
                current_estimate_ls = list()
                for t in range(T):
                    rgb_latent = rgb_latents[:, t]
                    noisy_latent = noisy_latents[:, t]
                    class_emb = self.pipeline.get_class_emb(class_id=1, bs=dev_bs, device=self.device)
                    model_pred = self.pipeline.single_forward(
                        rgb_latent,
                        noisy_latent,
                        timesteps,
                        unet_kwargs={"encoder_hidden_states": encoder_hidden_states, "class_labels": class_emb},
                    )
                    # Decode
                    current_latent_estimate = self.compute_latent_estimate(model_pred.sample, noisy_latent, timesteps)
                    current_latent_estimate_ls.append(current_latent_estimate)

                    current_estimate = self.pipeline.decode_latent(current_latent_estimate)    # B,4,h,w --> B,3,H,W
                    current_estimate_ls.append(current_estimate)

                # Stack for video or concat single image
                current_estimate = torch.stack(current_estimate_ls, dim=1)
                current_latent_estimate = torch.stack(current_latent_estimate_ls, dim=1)
                del current_estimate_ls, current_latent_estimate_ls
                torch.cuda.empty_cache()

                current_estimate_proj = {}
                for proj_name in self.proj_names:
                    flatten_inp = current_estimate.view(dev_bs*T, *current_estimate.shape[2:])          # B*T, C, H, W
                    proj_out = self.projectors[proj_name](flatten_inp)                              # B*T, N, H, W
                    current_estimate_proj[proj_name] = proj_out.view(dev_bs, T, *proj_out.shape[1:])    # B, T, N, H, W
                
                #### === Backward pass === ####
                loss_outs = {}
                loss = torch.tensor(0.0, device=self.device)
                for proj_name in self.proj_names:
                    outputs = (current_estimate_proj[proj_name], current_estimate, proj_name)  # projected output, rgb raw output
                    supervision = batch['supervision'][proj_name]
                    is_gt = torch.all(supervision == 2)
                    is_pseudo = torch.any(supervision == 1)
                    is_none = torch.all(supervision == 0)
                    if is_gt:
                        loss_task = "{}-gt".format(proj_name.split('-')[0])
                    elif is_pseudo or is_none:
                        loss_task = "{}-pseudo".format(proj_name.split('-')[0])

                    loss_out = self.loss_functions[loss_task](outputs, batch, self.device, self.weight_dtype)
                    if is_none:
                        # If no supervision, zero out all losses
                        loss_out = {k: v * 0.0 for k, v in loss_out.items()}
                    loss_outs[loss_task] = loss_out
                    loss += loss_out['total']

                for loss_item in self.loss_item_id.keys():
                    loss_task, loss_name = loss_item.split(':')
                    valid = loss_task in loss_outs and loss_name in loss_outs[loss_task]
                    if valid:
                        loss_val = loss_outs[loss_task][loss_name]
                    else:
                        loss_val = torch.zeros_like(loss)
                    
                    gathered_loss = self.accelerator.gather(loss_val).reshape(-1)
                    gathered_valid = torch.tensor(accelerate.utils.gather_object([valid]), device=self.device).bool()

                    filtered_loss = gathered_loss[gathered_valid]
                    if filtered_loss.numel() > 0:
                        train_loss_per_item[loss_item] += filtered_loss.sum().item()
                        train_cnt_per_item[loss_item] += filtered_loss.numel()
                
                train_loss += self.accelerator.gather(loss).reshape(-1).mean().item() / self.args.gradient_accumulation_steps

                # Backpropagate
                self.pipeline.prepare_backward()
                self.accelerator.wait_for_everyone()
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                    for proj_name, proj in self.projectors.items():
                        self.accelerator.clip_grad_norm_(proj.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # Checks if the self.accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                for loss_item in self.loss_item_id.keys():
                    self.accelerator.log({f"train_loss-{loss_item}": train_loss_per_item[loss_item] / train_cnt_per_item[loss_item]}, step=self.global_step)
                self.accelerator.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=self.global_step)

                if self.global_step % self.args.train_vis_steps == 0:
                    for ii in range(len(batch["rgb"])):
                        vis_train = get_training_vis(batch, current_estimate_proj, ii, self.proj_names, self.proj_color_codes)
                        self.accelerator.log({f"train_vis/pred_{ii}": wandb.Image(vis_train)}, step=self.global_step)

                        vis_latent = get_latent_vis(current_latent_estimate, batch, ii)
                        self.accelerator.log({f"train_vis/latent_{ii}": wandb.Image(vis_latent)}, step=self.global_step)
                    self.accelerator.wait_for_everyone()

                train_loss = 0.0
                train_loss_per_item = {item: 0.0 for item in self.loss_item_id.keys()}
                train_cnt_per_item = {item: 1e-6 for item in self.loss_item_id.keys()}

                # Save & evaluate model checkpoint
                if self.global_step % self.args.checkpointing_steps == 0:
                    self._save_model_checkpoint()

            # Log loss and learning rate for progress bar
            logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Break training
            if self.global_step >= self.args.max_train_steps:
                break    
 
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            self._save_final_model()

        self.logger.info(f"Finished training.")
        self.accelerator.end_training()


    def compute_latent_estimate(self, model_pred, noisy_latent, timesteps):
        """ Convert parameterized prediction into latent prediction.
        """
        # Get noise scheduling parameters for later conversion from a parameterized prediction into latent.
        noise_scheduler = self.pipeline.scheduler
        alpha_prod = noise_scheduler.alphas_cumprod.to(self.device, dtype=self.weight_dtype)
        beta_prod  = 1 - alpha_prod

        # Convert parameterized prediction into latent prediction. (DDIM code from diffusers)
        # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
        alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
        beta_prod_t  =  beta_prod[timesteps].view(-1, 1, 1, 1)
        if noise_scheduler.config.prediction_type == "v_prediction":
            latent_estimate = (alpha_prod_t**0.5) * noisy_latent - (beta_prod_t**0.5) * model_pred
        elif noise_scheduler.config.prediction_type == "epsilon":
            latent_estimate = (noisy_latent - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
        elif noise_scheduler.config.prediction_type == "sample":
            latent_estimate = model_pred
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        # clip and threshold prediction (only here for completeness, not used by SD2 or our models with v_prediction)
        if noise_scheduler.config.thresholding:
            pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
        elif noise_scheduler.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp( -noise_scheduler.config.clip_sample_range, noise_scheduler.config.clip_sample_range)
        
        return latent_estimate


    def _save_final_model(self):
        # Create SD pipeline using the trained modules and save it.
        unet = self.unwrap_model(self.unet)
        scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_local_path, 
            subfolder="scheduler", 
            timestep_spacing="trailing", # set scheduler timestep spacing to trailing for later inference.
            revision=None, 
            variant=None
        )
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.args.pretrained_model_local_path,
            text_encoder=self.pipeline.text_encoder,
            vae=self.pipeline.vae,
            unet=unet,
            scheduler=scheduler,
            revision=None,
            variant=None,
        )
        self.logger.info(f"Saving pipeline to {self.args.output_dir}")
        pipeline.save_pretrained(self.args.output_dir)
        for proj_name, projector in self.projectors.items():
            projector = self.unwrap_model(projector)
            projector.save_pretrained(osp.join(self.args.output_dir, "projectors"))


    def _save_model_checkpoint(self):
        """ Save the model checkpoint at the current global step.
        """
        self.logger.info(f"Entered Saving Code at global step {self.global_step} checkpointing_steps {self.args.checkpointing_steps}")
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if self.accelerator.is_main_process:
            if self.args.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.args.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                if len(checkpoints) >= self.args.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.args.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    self.logger.info(
                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                    )
                    self.logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                    for removing_checkpoint in removing_checkpoints:
                        ckpt_idx = int(removing_checkpoint.strip('checkpoint-'))
                        if self.args.checkpoints_keep_freq > 0 and (ckpt_idx % self.args.checkpoints_keep_freq == 0):
                            continue
                        removing_checkpoint = osp.join(self.args.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
        self.accelerator.wait_for_everyone()

        save_path = osp.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        for proj_name, projector in self.projectors.items():
            projector = self.unwrap_model(projector)
            msg = projector.save_pretrained(osp.join(save_path, "projectors"))
            self.logger.info(msg)
        self.logger.info(f"Saved state to {save_path}")


    def _get_task_encodings(self):
        """ Get task encodings for the text encoder.
        """
        task_to_text_prompt = "video_dense_pixel_prediction"
        task_token    = self.pipeline.tokenizer([task_to_text_prompt], padding="max_length", truncation=True, return_tensors="pt").input_ids
        task_token    = task_token.to(self.device)
        task_encoding = self.pipeline.text_encoder(task_token, return_dict=False)[0]
        task_encoding = task_encoding.to(self.device)
        task_encodings = task_encoding

        return task_encodings, task_to_text_prompt


    def _setup_loss_functions(self):
        """ Setup the loss functions for the UNID model.
        """
        loss_functions = {}
        loss_item_id = {}
        for loss, loss_info in self.args.losses.items():
            if loss_info.get('copy') is not None:
                copy_loss = loss_info.pop('copy')
                self.args.losses[loss] = self.args.losses[copy_loss].copy()
        
        for loss, loss_info in self.args.losses.items():
            loss_name = loss_info.cls_name
            loss_class = getattr(__import__('loss', fromlist=[loss_name]), loss_name)
            if loss.endswith('-pseudo'):
                loss_info.multiplier *= self.args.additional_pseudo_multiplier
            loss_functions[loss] = loss_class(**loss_info.params)
            for loss_name in loss_info.params.keys():
                loss_item = f'{loss}:{loss_name}'
                assert loss_item not in loss_item_id
                loss_item_id[loss_item] = len(loss_item_id)
    
        return loss_functions, loss_item_id


    def _init_train_logging(self):
        """ Initialize logging for the training process.
        """
        total_batch_size = self.args.batch_size_per_gpu * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size_per_gpu}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        self.logger.info(f"  Args: {self.args}")


    def resume_from_checkpoint(self):
        """ Resume training from a checkpoint if specified.
        """
        if self.args.resume_from_checkpoint != "latest":
            path = osp.basename(self.args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            self.accelerator.print(
                f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            self.args.resume_from_checkpoint = None
        else:  
            self.accelerator.print(f"Resuming from checkpoint {path}")
            self.accelerator.load_state(osp.join(self.args.output_dir, path))
            self.global_step = int(path.split("-")[1])


    def unwrap_model(self, model):
        """ Function for unwrapping if model was compiled with `torch.compile`.
        """
        model = self.accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    def _setup_tracker(self):
        """ Setup the tracker for logging and monitoring the training process.
        """
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        tracker_config = OmegaConf.to_container(self.args, resolve=True)
        for key, value in tracker_config.items():
            if not (isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool)):
                tracker_config[key] = str(value) # eliminate list in config for saving
        self.accelerator.init_trackers(self.args.tracker_project_name, tracker_config, init_kwargs={"wandb": {"name": osp.basename(self.args.output_dir)}})


    def _get_weights_dtype(self):
        """ Setup the weight dtype based on the mixed precision setting.
        """
        # Mixed precision and weight dtype
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            self.args.mixed_precision = self.accelerator.mixed_precision
            self.unet.to(dtype=weight_dtype)
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            self.args.mixed_precision = self.accelerator.mixed_precision
            self.unet.to(dtype=weight_dtype)
        self.pipeline.vae.to(self.device, dtype=weight_dtype)    
        self.pipeline.text_encoder.to(self.device, dtype=weight_dtype)
        return weight_dtype
    

    def _setup_projectors(self):
        """ Setup the RGB projectors for the UNID model.
        """
        projectors = {}
        proj_names = []
        input_conv = self.pipeline.dec_dim
        self.proj_color_codes = {}

        for proj_name, proj_info in self.args.projectors.items():
            output_dim = proj_info.proj_dim
            hidden_dims = proj_info.hidden_dims
            projectors[proj_name] = TaskProjector(
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                name=proj_name,
                input_conv=input_conv,
            )
            proj_names.append(proj_name)
            self.logger.info(f'Created projector {proj_name} with input conv {input_conv} | output dim {output_dim} | hidden dims {hidden_dims}.')
            if proj_name.startswith('semseg'):
                train_datasets = self.train_dataloader.task_datasets
                found = None
                for task, datasets_list in train_datasets.items():
                    for dataset in datasets_list['datasets']:
                        if proj_name in dataset.gt_tasks:
                            found = dataset.sem_color_code
                            self.logger.info(f'Loaded semseg color codes for projector {proj_name} from dataset {dataset.dataset_name}.')
                if found is None:
                    raise Exception('No dataset found.')
                self.proj_color_codes[proj_name] = found
            elif proj_name.startswith('partseg'):
                codes = torch.rand(16,3); codes[0,:] = 0.0; codes[15,:] = 1.0
                self.proj_color_codes[proj_name] = codes
            elif proj_name.startswith('matseg'):
                train_datasets = self.train_dataloader.task_datasets
                found = None
                for task, datasets_list in train_datasets.items():
                    for dataset in datasets_list['datasets']:
                        if proj_name in dataset.gt_tasks:
                            found = dataset.mat_color_code
                            self.logger.info(f'Loaded semseg color codes for projector {proj_name} from dataset {dataset.dataset_name}.')
                if found is None:
                    raise Exception('No dataset found.')
                self.proj_color_codes[proj_name] = found

        return torch.nn.ModuleDict(projectors), proj_names


    def _setup_train_loader(self):
        """ Setup the training dataloader for the UNID model.
        """
        train_dataloader = UnifiedDataLoader(
            config=self.args.tasks,
            batch_size=self.args.batch_size_per_gpu,
            temporal_length=self.args.get("temporal_length", 1),  # Default to 1 if not specified
            num_workers_per_dataset=self.args.dataloader_num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_dataloader


    def _setup_optimizer(self):
        """ Setup the optimizer for the UNet model.
        """
        opt_input = [{
            'params': self.unet.parameters(),
            'lr': self.args.learning_rate
        }] + [{
            'params': projector.parameters(),
            'lr': self.args.learning_rate * self.args.projectors[proj_name].get('lr_multiplier', 1.0)
        } for proj_name, projector in self.projectors.items()]
        optimizer = AdamW(
            opt_input,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        return optimizer


    def _setup_lr_scheduler(self):
        """ Setup the learning rate scheduler for the optimizer.
        """
        lr_func_cls = {
            "linear": IterLinear,
            "exponential": IterExponential,
            "constant": IterConstant,
        }[self.args.lr_scheduler_type]
        lr_func = lr_func_cls(
            total_iter_length = self.args.lr_total_iter_length * self.accelerator.num_processes,
            final_ratio = 0.01,
            warmup_steps = self.args.lr_exp_warmup_steps * self.accelerator.num_processes
        )
        lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
        return lr_scheduler


    def _setup_pipeline(self):
        """ Load model components and create the UNID pipeline.
        """
        unet            = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_local_path, subfolder="unet", revision=None)
        vae             = AutoencoderKL.from_pretrained(self.args.pretrained_model_local_path, subfolder="vae", revision=None, variant=None)
        noise_scheduler = DDPMScheduler.from_pretrained(self.args.pretrained_model_local_path, subfolder="scheduler")
        text_encoder    = CLIPTextModel.from_pretrained(self.args.pretrained_model_local_path, subfolder="text_encoder", revision=None, variant=None)
        tokenizer       = CLIPTokenizer.from_pretrained(self.args.pretrained_model_local_path, subfolder="tokenizer", revision=None)
        
        unid_pipeline = UnidPipeline(
            unet=unet,
            vae=vae,
            scheduler=noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            architecture=self.args.architecture,
        )

        msg = unid_pipeline.build_model(
            is_train=True, 
            noise_type=self.args.noise_type, 
            enable_xformers=self.args.enable_xformers_memory_efficient_attention
        )
        [self.logger.info(m) for m in msg]
        
        # Freeze or set model components to training mode
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.train()

        # Diffusers model loading and saving functions 
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_local_main_process:
                    for i, model in enumerate(models):
                        if type(model) == UNet2DConditionModel:
                            model.save_pretrained(osp.join(output_dir, "unet"))
                        elif type(model) == TaskProjector:
                            model.save_pretrained(osp.join(output_dir, "projectors"))
                        else:
                            raise ValueError(f"Unknown model type {type(model)} for saving.")
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            def load_model_hook(models, input_dir):
                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if type(model) == UNet2DConditionModel:
                        # load diffusers style into model
                        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                        model.register_to_config(**load_model.config)
                        # model.load_state_dict(load_model.state_dict())
                        load_state = safetensors.torch.load_file(os.path.join(input_dir, 'unet', 'diffusion_pytorch_model.safetensors'))
                        model.load_state_dict(load_state, strict=True)
                        del load_model
                        self.logger.info(f"Loaded UNet successfully from {input_dir}")
                    elif type(model) == TaskProjector:
                        model.from_pretrained(osp.join(input_dir, "projectors"))
                        self.logger.info(f"Loaded Projector {model.name} successfully from {input_dir}")
                    else:
                        raise ValueError(f"Unknown model type {type(model)} for loading.")
                    
            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        # Gradient checkpointing
        if self.args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        return unid_pipeline


    def _setup_pretrained_model(self, pretrained_config_path="training/configs/pretrained_models.yaml"):
        """Check that the locally supplied base model is present."""
        # Load pretrained models config
        with open(pretrained_config_path, "r") as f:
            pretrained_models_config = yaml.safe_load(f)

        self.args.pretrained_model_local_path = pretrained_models_config[self.args.pretrained_model_name]["local_path"]
        if not osp.isdir(self.args.pretrained_model_local_path):
            raise FileNotFoundError(
                f"Base model directory is missing: {self.args.pretrained_model_local_path}. "
                "Obtain the required Stable Diffusion 2 Diffusers files, place them at this path, "
                "or update training/configs/pretrained_models.yaml. No model is downloaded automatically."
            )


    def _setup_logging_accelerator(self):
        """ Setup logging and accelerator for distributed training.
        """
        logger = get_logger(__name__, log_level="INFO")
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=osp.join(self.args.output_dir, self.args.logging_dir)
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_config=accelerator_project_config,
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        # Save training arguments in a .txt file
        if accelerator.is_local_main_process:
            args_dict = vars(self.args)
            args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
            args_str += f'\nnum_processes: {accelerator.num_processes}'
            args_str += f'\ndistributed_type: {accelerator.distributed_type}'
            args_path = osp.join(self.args.output_dir, "arguments.txt")
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(args_path, 'w') as file:
                file.write(args_str)
        
        return logger, accelerator


    def _check_args_conditions(self):
        """ Check conditions on the arguments to ensure they are valid.
        """
        assert self.args.tasks, "No task is provided. Set --tasks."
        if self.args.noise_type is None:
            self.logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")


    def _load_config(self, config_path, ow_args):
        """Load training configuration from YAML file using OmegaConf"""
        if not osp.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
        
        # Load config using OmegaConf
        config = OmegaConf.load(config_path)
        
        # Calculate gradient_accumulation_steps based on GPU count if not specified
        if not hasattr(config, "gradient_accumulation_steps") or config.gradient_accumulation_steps is None:
            num_gpus = torch.cuda.device_count()
            config.gradient_accumulation_steps = max(1, config.effective_batch_size // (config.batch_size_per_gpu * num_gpus))
        
        overwrite_args(config, ow_args)
        return config

def get_parser():
    parser = argparse.ArgumentParser(description="training config overwrites")
    parser.add_argument("config_path", type=str, help="Path to the YAML config file for training.")
    parser.add_argument("--overwrite", nargs='*', help="List of config values to overwrite in the format key=value. For nested keys, use dot notation (e.g., model.learning_rate=0.001).")
    return parser

if __name__ == "__main__":
    # Load config from command line argument
    args = get_parser().parse_args()
    trainer = Trainer(args.config_path, args.overwrite)
    trainer.train()

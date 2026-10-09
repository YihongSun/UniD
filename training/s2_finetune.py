# @GonzaloMartinGarcia
# Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.
# This training code is a modified version of the original text-to-image SD training code from the HuggingFace Inc. Team,
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py.
  
import logging
import math
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
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors
import argparse
import yaml

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
from s0_specialist import Trainer as SpecialistTrainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unid import UnidPipeline
from unid.util.rgb_projector import TaskProjector
from unid.util.dpt_connector import DPTConnector

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.27.0.dev0")

class Trainer(SpecialistTrainer):
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
        self.connectors, self.cnct_names = self._setup_connectors()   # FT: added
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        
        # Prepare everything with `self.accelerator` (Move to GPU)
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        for proj_name, projector in self.projectors.items():
            self.projectors[proj_name] = self.accelerator.prepare(self.projectors[proj_name])
        for cnct_name in self.cnct_names:   # FT: added
            self.connectors[cnct_name] = self.accelerator.prepare(self.connectors[cnct_name])
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
                current_estimate_ls = list()
                for t in range(T):
                    rgb_latent = rgb_latents[:, t]
                    noisy_latent = noisy_latents[:, t]
                    class_emb = self.pipeline.get_class_emb(class_id=1, bs=dev_bs, device=self.device)

                    self.pipeline._set_store_features(True)
                    with torch.no_grad():
                        model_pred, interm_feats = self.pipeline.single_forward(
                            rgb_latent,
                            noisy_latent,
                            timesteps,
                            unet_kwargs={"encoder_hidden_states": encoder_hidden_states, "class_labels": class_emb},
                            return_intermediate_feats=True
                        )
                    self.pipeline._set_store_features(False)
                    # Decode
                    cnct_sample = self.connectors[self.args.task_to_finetune](interm_feats)   # FT: added
                    current_latent_estimate = self.compute_latent_estimate(cnct_sample, noisy_latent, timesteps)

                    current_estimate = self.pipeline.decode_latent(current_latent_estimate)    # B,4,h,w --> B,3,H,W
                    current_estimate_ls.append(current_estimate)

                # Stack for video or concat single image
                current_estimate = torch.stack(current_estimate_ls, dim=1)
                del current_estimate_ls
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
                        self.accelerator.log({f"train_vis/{ii}": wandb.Image(vis_train)}, step=self.global_step)
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
        for cnct_name, connector in self.connectors.items():   # FT: added
            connector = self.unwrap_model(connector)
            connector.save_pretrained(osp.join(self.args.output_dir, "connectors"))



    def _setup_tracker(self):
        """ Setup the tracker for logging and monitoring the training process.
        """
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        tracker_config = OmegaConf.to_container(self.args, resolve=True)
        for key, value in tracker_config.items():
            if not (isinstance(value, int) or isinstance(value, float) or isinstance(value, str) or isinstance(value, bool)):
                tracker_config[key] = str(value) # eliminate list in config for saving
        self.accelerator.init_trackers(self.args.tracker_project_name, tracker_config, init_kwargs={"wandb": {"name": osp.basename(self.args.output_dir)+'_'+self.args.task_to_finetune}})


    def _setup_projectors(self):
        """ Setup the RGB projectors for the UNID model.
        """
        task2ft = self.args.task_to_finetune       # FT: modified
        projectors = {k:v for k,v in self.pipeline.projectors.items() if k.startswith(task2ft)}
        proj_names = list(projectors.keys())

        self.proj_color_codes = {}
        for proj_name in proj_names:
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
    
    def _setup_connectors(self):
        """ Setup the connectors for the UNID model.
        """
        task2ft = self.args.task_to_finetune       # FT: added
        connectors = {k:v for k,v in self.pipeline.connectors.items() if k.startswith(task2ft)}
        conn_names = list(connectors.keys())
        return torch.nn.ModuleDict(connectors), conn_names


    def _setup_optimizer(self):
        """ Setup the optimizer for the UNet model.
        """
        opt_input = [{
            'params': itertools.chain(*[module.parameters() for module in self.projectors.values()]),      # FT: added
            'lr': self.args.learning_rate
        }, {
            'params': itertools.chain(*[module.parameters() for module in self.connectors.values()]),      # FT: added
            'lr': self.args.learning_rate
        }]
        optimizer = AdamW(
            opt_input,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        return optimizer


    def _setup_pipeline(self):
        """ Load model components and create the UNID pipeline.
        """
        unet            = UNet2DConditionModel.from_pretrained('checkpoints/pretrained_sd2', subfolder="unet", revision=None)
        vae             = AutoencoderKL.from_pretrained('checkpoints/pretrained_sd2', subfolder="vae", revision=None, variant=None)
        noise_scheduler = DDPMScheduler.from_pretrained('checkpoints/pretrained_sd2', subfolder="scheduler")
        text_encoder    = CLIPTextModel.from_pretrained('checkpoints/pretrained_sd2', subfolder="text_encoder", revision=None, variant=None)
        tokenizer       = CLIPTokenizer.from_pretrained('checkpoints/pretrained_sd2', subfolder="tokenizer", revision=None)
        
        # FT: reworked
        unid_pipeline, msg = UnidPipeline.from_pretrained(
            pretrained_model_name_or_path = self.args.model_to_finetune,
            unet=unet, 
            vae=vae, 
            scheduler=noise_scheduler, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer, 
            is_train=True,
            noise_type=self.args.noise_type,
        )
        [self.logger.info(m) for m in msg]
        
        # Freeze or set model components to training mode
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)

        # Diffusers model loading and saving functions 
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            def save_model_hook(models, weights, output_dir):
                if self.accelerator.is_local_main_process:
                    for i, model in enumerate(models):
                        if type(model) == UNet2DConditionModel:
                            pass # skip saving since it is frozen
                        elif type(model) == TaskProjector:
                            model.save_pretrained(osp.join(output_dir, "projectors"))
                        elif type(model) == TaskConnector or type(model) == DPTConnector:
                            model.save_pretrained(osp.join(output_dir, "connectors"))
                        else:
                            raise ValueError(f"Unknown model type {type(model)} for saving.")
                        # make sure to pop weight so that corresponding model is not saved again
                        weights.pop()
            def load_model_hook(models, input_dir):
                for _ in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()
                    if type(model) == UNet2DConditionModel:
                        continue # skip loading since it is frozen
                    elif type(model) == TaskProjector:
                        model.from_pretrained(osp.join(input_dir, "projectors"))
                        self.logger.info(f"Loaded Projector {model.name} successfully from {input_dir}")
                    elif type(model) == TaskConnector or type(model) == DPTConnector:
                        model.from_pretrained(osp.join(input_dir, "connectors"))
                        self.logger.info(f"Loaded Connector successfully from {input_dir}")
                    else:
                        raise ValueError(f"Unknown model type {type(model)} for loading.")
                    
            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        # Gradient checkpointing
        if self.args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        return unid_pipeline


    def _check_args_conditions(self):
        """ Check conditions on the arguments to ensure they are valid.
        """
        assert self.args.tasks, "No task is provided. Set --tasks."
        if self.args.noise_type is None:
            self.logger.warning("Noise type is `None`. This setting is only meant for checkpoints without image conditioning, such as Stable Diffusion.")
        
        ### Setting args.output_dir based on model_to_finetune - FT: added
        assert 'unified' in self.args.model_to_finetune
        self.args.output_dir = self.args.model_to_finetune.replace('unified', 'ft_unified') + self.args.experiment_name


    def _load_config(self, config_path, ow_args):
        """Load training configuration from YAML file using OmegaConf"""
        if not osp.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
        
        # Load config using OmegaConf
        config = OmegaConf.load(config_path)
        overwrite_args(config, ow_args)
        
        # Calculate gradient_accumulation_steps based on GPU count if not specified
        if not hasattr(config, "gradient_accumulation_steps") or config.gradient_accumulation_steps is None:
            num_gpus = torch.cuda.device_count()
            config.gradient_accumulation_steps = max(1, config.effective_batch_size // (config.batch_size_per_gpu * num_gpus))
        
        # FT: load distillation args (used during inference)
        with open(osp.join(config.model_to_finetune, "arguments.txt"), 'r') as f:
            dist_arg = [l.strip() for l in f.readlines()]
            dist_arg = eval([l for l in dist_arg if l.startswith("_content:")][0].replace("_content:", "").strip())
        config.task_to_train = dist_arg['task_to_train']
        config.distill_tasks = dist_arg['distill_tasks']
        config.connectors = dist_arg['connectors']

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

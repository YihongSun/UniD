import logging
import math
import subprocess
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
import torch.nn.functional as F
import torch.distributed as dist
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import safetensors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from util.utils import proj_pca, proj_pca_precomp
from s0_specialist import Trainer as SpecialistTrainer
import yaml

from unid import UnidPipeline
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
    
    def __init__(self, config_path):
        """ Initialize the Trainer with the configuration file path.
        """
        
        self.args = self._load_config(config_path)
        self._check_args_conditions()

        self.logger, self.accelerator = self._setup_logging_accelerator()
        if self.args.seed is not None:
            set_seed(self.args.seed, device_specific=True)
        
        self.train_dataloader = self._setup_train_loader()
        self._setup_pretrained_model()
        self.pipeline = self._setup_pipeline()
        self.unet = self.pipeline.unet
        self.connectors, self.cnct_names = self._setup_connectors()
        self.optimizer = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()
        
        # Prepare everything with `self.accelerator` (Move to GPU)
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        for cnct_name in self.cnct_names:
            self.connectors[cnct_name] = self.accelerator.prepare(self.connectors[cnct_name])
        self.device       = self.accelerator.device
        self.weight_dtype = self._get_weights_dtype()
        self.distill_src_unets = self._setup_distill_src_unets()

        if self.accelerator.is_main_process:
            self._setup_tracker()

        self.task_to_tid = {task: i for i, task in enumerate(self.args.tasks.keys())}
        self.loss_functions, self.loss_item_id = self._setup_loss_functions()
        self.task_encodings, self.task_to_text_prompt = self._get_task_encodings()

        ## Distillation specific
        self.pipeline.vae.decoder.to("cpu")
        del self.pipeline.vae.decoder
        torch.cuda.empty_cache()
    
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

                #### === Forward pass - Compute distillation targets === ####
                for cnct_name in self.cnct_names:
                    if cnct_name not in self.distill_src_unets:
                        assert cnct_name == "restore", "Distillation target UNet not found for task {}".format(cnct_name)
                        batch[cnct_name] = rgb_latents  # restoration target
                        continue
                    with torch.no_grad():
                        src_unet = self.distill_src_unets[cnct_name]
                        self.pipeline.init(is_video=batch["is_video"], is_train=False, unet=src_unet)
                        distill_tgt = list()
                        for t in range(T):
                            rgb_latent = rgb_latents[:, t]
                            noisy_latent = noisy_latents[:, t]
                            model_pred = self.pipeline.single_forward(
                                rgb_latent,
                                noisy_latent,
                                timesteps,
                                unet_kwargs={"encoder_hidden_states": encoder_hidden_states},
                                unet=src_unet
                            )
                            # Decode
                            distill_tgt.append(self.compute_latent_estimate(model_pred.sample, noisy_latent, timesteps))
                        batch[cnct_name] = torch.stack(distill_tgt, dim=1)  # B, T, 4, h, w

                #### === Forward pass - Compute Unified features === ####
                input_bt_flat = batch["input"].view(dev_bs*T, *batch["input"].shape[2:])  # B*T, C, H, W
                input_latents = self.pipeline.encode_rgb(input_bt_flat.to(self.device, dtype=self.weight_dtype))
                input_latents = input_latents.view(dev_bs, T, *input_latents.shape[1:])  # B, T, 4, h, w

                self.pipeline.init(is_video=batch["is_video"], is_train=True)
                distill_pred = {conn: [] for conn in self.cnct_names}
                for t in range(T):
                    input_latent = input_latents[:, t]
                    noisy_latent = noisy_latents[:, t]
                    class_emb = self.pipeline.get_class_emb(class_id=1, bs=dev_bs, device=self.device)

                    self.pipeline._set_store_features(True)
                    model_pred, interm_feats = self.pipeline.single_forward(
                        input_latent,
                        noisy_latent,
                        timesteps,
                        unet_kwargs={"encoder_hidden_states": encoder_hidden_states, "class_labels": class_emb},
                        return_intermediate_feats=True
                    )
                    self.pipeline._set_store_features(False)

                    for cnct_name in self.cnct_names:
                        cnct_sample = self.connectors[cnct_name](interm_feats)
                        distill_pred[cnct_name].append(
                            self.compute_latent_estimate(cnct_sample, noisy_latent, timesteps)
                        )
                
                for cnct_name, p in distill_pred.items():
                    distill_pred[cnct_name] = torch.stack(p, dim=1)  # B, T, N, H, W
                torch.cuda.empty_cache()

                #### === Forward pass - Compute losses === ####
                loss_outs = {}
                loss = torch.tensor(0.0, device=self.device)
                for cnct_name in self.cnct_names:
                    outputs = (distill_pred[cnct_name], cnct_name)
                    loss_out = self.loss_functions[cnct_name](outputs, batch, self.device, self.weight_dtype)

                    loss_outs[cnct_name] = loss_out
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

                #### === Backward pass - Backpropagate === ####
                self.pipeline.prepare_backward()
                self.accelerator.wait_for_everyone()
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), self.args.max_grad_norm)
                    for cnct_name, connector in self.connectors.items():
                        self.accelerator.clip_grad_norm_(connector.parameters(), self.args.max_grad_norm)
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
                    vis_pred = self.get_connector_vis(distill_pred, batch, 0)
                    for ii in range(len(vis_pred)):
                        self.accelerator.log({f"train_vis/{ii}": wandb.Image(vis_pred[ii])}, step=self.global_step)
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
    
    def get_connector_vis(self, distill_pred, batch, ii):
        """ Get visualization for the connector predictions.
        """

        H, W = batch["input"].shape[3], batch["input"].shape[4]
        v = [torch.cat([_ for _ in batch["input"][ii]], dim=2).to(self.device),]  # inp_img C, H, W*T, 
        for cnct_name in self.cnct_names:
            pred_cnct = distill_pred[cnct_name][ii]    # T, N, H, W
            gt_cnct   = batch[cnct_name][ii]          # T, N, H, W

            gt_vis, (mu,V) = proj_pca(gt_cnct, target_dim=3)    # T, 3, H, W
            pred_vis = proj_pca_precomp(pred_cnct, mu, V)       # T, 3, H, W

            gt_vis = F.interpolate(gt_vis, size=(H, W), mode='nearest')
            pred_vis = F.interpolate(pred_vis, size=(H, W), mode='nearest')

            v.append( torch.cat([_ for _ in pred_vis], dim=2) )  # pred_vis C, H, W*T
            v.append( torch.cat([_ for _ in gt_vis], dim=2) )    # gt_vis C, H, W*T

        vis = torch.cat(v, dim=1).permute(1,2,0)
        vis_all = self.accelerator.gather(vis).reshape(-1, *vis.shape)  # (num_processes, H, W, C)

        vis_np = [((x.detach().cpu().numpy() + 1) / 2.0 * 255).astype(np.uint8) for x in vis_all]
        return vis_np


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
        for cnct_name, connector in self.connectors.items():
            connector = self.unwrap_model(connector)
            connector.save_pretrained(osp.join(self.args.output_dir, "connectors"))
        
        # run subprocess on bash scripts/combine_ckpts.py -c {self.args.output_dir} to combine the saved checkpoint into one directory for easier loading later.
        # This is needed because the current accelerator checkpointing saves the unet and connectors separately, which can be cumbersome to load later for inference.
        combine_script_path = osp.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "combine_ckpts.py")
        subprocess.run(["python3", combine_script_path, "-c", self.args.output_dir])

    def _save_model_checkpoint(self):
        """ Save the model checkpoint at the current global step.
        """
        self.logger.info(f"Entered Saving Code at global step {self.global_step} checkpointing_steps {self.args.checkpointing_steps}")
        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
        if self.accelerator.is_local_main_process:
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
        # Ensure all processes wait for cleanup to finish before saving
        self.accelerator.wait_for_everyone()

        save_path = osp.join(self.args.output_dir, f"checkpoint-{self.global_step}")
        self.accelerator.save_state(save_path)
        self.logger.info(f"Saved state to {save_path}")


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
            loss_functions[loss] = loss_class(**loss_info.params)
            for loss_name in loss_info.params.keys():
                loss_item = f'{loss}:{loss_name}'
                assert loss_item not in loss_item_id
                loss_item_id[loss_item] = len(loss_item_id)
    
        return loss_functions, loss_item_id
    

    def _setup_connectors(self):
        """ Setup the RGB connectors for the UNID model.
        """
        connectors = {}
        cnct_names = []
        assert self.args.connectors.cls_name == 'DPTConnector'
        cnct_conf = self.args.connectors.config
        for cnct_name in self.args.distill_tasks.keys():
            connectors[cnct_name] = DPTConnector(**cnct_conf, name=cnct_name)
            cnct_names.append(cnct_name)
            self.logger.info('Created DPTConnector({}) with {}'.format(
                cnct_name, ', '.join([f'{k}:{v}' for k,v in cnct_conf.items()])
            ))
        return torch.nn.ModuleDict(connectors), cnct_names


    def _setup_optimizer(self):
        """ Setup the optimizer for the UNet model.
        """
        opt_input = [{
            'params': self.unet.parameters(),
            'lr': self.args.learning_rate
        }, {
            'params': itertools.chain(*[module.parameters() for module in self.connectors.values()]),
            'lr': self.args.learning_rate * self.args.connectors.lr_multiplier
        }]
        optimizer = AdamW(
            opt_input,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )
        return optimizer


    def _setup_distill_src_unets(self):
        """ Setup the source UNet models for distillation.
        """
        rank, world_size = self.accelerator.process_index, self.accelerator.num_processes
        self.logger.info(f"Number of processes: {world_size}  |  distill src unets: {self.cnct_names}")

        distill_src_unets = {}
        for ii, distill_task in enumerate(self.cnct_names):
            distill_dir = self.args.distill_tasks[distill_task]['ckpt']
            if distill_dir is None:
                self.logger.info(f"  No distillation source UNet for task {distill_task}.")
                continue
            ## get architecture from distill dir
            arg_txt = os.path.join(distill_dir, "arguments.txt")
            if not os.path.exists(arg_txt):
                arg_txt = os.path.join(osp.dirname(distill_dir), "arguments.txt")
            with open(arg_txt, "r") as f:
                arguments = [l.strip() for l in f.readlines() if l.startswith]
                arg = eval([l for l in arguments if l.startswith("_content:")][0].replace("_content:", "").strip())
                load_arch = arg["architecture"]

            # Create placeholder pipeline
            unet            = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_local_path, subfolder="unet", revision=None)
            vae             = AutoencoderKL.from_pretrained(self.args.pretrained_model_local_path, subfolder="vae", revision=None, variant=None)
            placeholder_pipeline = UnidPipeline(unet=unet, vae=vae, scheduler=None, text_encoder=None, tokenizer=None, architecture=load_arch)
            placeholder_pipeline.build_model(
                is_train=True, 
                noise_type=self.args.noise_type, 
                enable_xformers=self.args.enable_xformers_memory_efficient_attention
            )
            src_unet = placeholder_pipeline.unet
            load_state = safetensors.torch.load_file(os.path.join(distill_dir, 'unet', 'diffusion_pytorch_model.safetensors'))
            src_unet.load_state_dict(load_state, strict=True)
            src_unet.to(self.device)
            src_unet.eval()
            for param in src_unet.parameters():
                param.requires_grad = False
            distill_src_unets[distill_task] = src_unet
            del vae, placeholder_pipeline
            self.logger.info(f"  Loaded distillation source UNet for task {distill_task} from [{distill_dir}]")
        
        return distill_src_unets

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
                        elif type(model) == DPTConnector:
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
                        load_state = safetensors.torch.load_file(os.path.join(input_dir, 'unet', 'diffusion_pytorch_model.safetensors'))
                        model.load_state_dict(load_state, strict=True)
                        # del load_model
                        self.logger.info(f"Loaded UNet successfully from {input_dir}")
                    elif type(model) == DPTConnector:
                        model.from_pretrained(osp.join(input_dir, "connectors"))
                        self.logger.info(f"Loaded {type(model)} {model.name} successfully from {input_dir}")
                    else:
                        raise ValueError(f"Unknown model type {type(model)} for loading.")
                    
            self.accelerator.register_save_state_pre_hook(save_model_hook)
            self.accelerator.register_load_state_pre_hook(load_model_hook)

        # Gradient checkpointing
        if self.args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        
        return unid_pipeline


    def _load_config(self, config_path):
        """Load training configuration from YAML file using OmegaConf"""
        if not osp.exists(config_path):
            raise ValueError(f"Config file {config_path} does not exist")
        
        # Load config using OmegaConf
        config = OmegaConf.load(config_path)
        
        # Calculate gradient_accumulation_steps based on GPU count if not specified
        if not hasattr(config, "gradient_accumulation_steps") or config.gradient_accumulation_steps is None:
            world_size = accelerate.state.PartialState().num_processes
            config.gradient_accumulation_steps = max(1, config.effective_batch_size // (config.batch_size_per_gpu * world_size))
            
        return config


if __name__ == "__main__":
    # Load config from command line argument
    if len(sys.argv) < 2:
        print("Error: Please provide a YAML config file path")
        print("Usage: python train.py config.yaml")
        sys.exit(1)
    
    trainer = Trainer(config_path=sys.argv[1])
    trainer.train()

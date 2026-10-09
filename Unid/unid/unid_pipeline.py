# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# @GonzaloMartinGarcia
# This file is a modified version of the original Marigold pipeline file. 
# Based on GeoWizard, we added the option to sample surface normals, marked with # add.

from typing import Dict, Union, Optional
import os, math, random
import os.path as osp
import numpy as np
from PIL import Image
import safetensors
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.utils import BaseOutput
from diffusers.models.attention_processor import XFormersAttnProcessor
import torch
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from .util.image_util import (
    chw2hwc,
    colorize_depth_maps,
    get_tv_resample_method,
    resize_max_res,
)
from .util.build_model import build_model_MemAttnTemp, replace_unet_conv_in
from .util.rgb_projector import TaskProjector
from .util.dpt_connector import DPTConnector
from .util.post_crf import apply_crf

class UnidOutput(BaseOutput):
    """
    Output class for Unified Perception pipeline.
    """
    projected: Dict[str, torch.Tensor]
    pred: torch.Tensor


class UnidPipeline(DiffusionPipeline):
    """
    Pipeline for Unified Perception.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: Union[DDIMScheduler,DDPMScheduler,LCMScheduler],
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        architecture: str = "BaseImg",  # allowing different architectures
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.arch = architecture
        self.text_prompt, self.text_embed = None, None
        self.projectors, self.proj_names = dict(), list()
        self.connectors, self.cnct_names = dict(), list()
        self.dec_dim = None

    ###### ====== Main inference function ====== ######
    @torch.no_grad()
    def __call__(
        self,
        input_image: Union[Image.Image, torch.Tensor],
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        noise="gaussian",
        text_prompt="",
        post_crf: bool = False,
    ) -> UnidOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            noise (`str`, *optional*, defaults to `gaussian`):
                Type of noise to be used for the initial depth map.
                Can be one of `gaussian`, `pyramid`, `zeros`.
            text_prompt (`str`, *optional*, defaults to ``):
                Text prompt for denoiser.
            post_crf (`bool`, *optional*, defaults to `False`):
                Whether to apply CRF post-processing for segmentation tasks. Only valid if the pipeline has segmentation projectors.
        """

        assert processing_res >= 0

        resample_method: InterpolationMode = get_tv_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------

        # Convert input to torch tensor
        if isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
            rgb = pil_to_tensor(input_image) # [H, W, rgb] -> [rgb, H, W]
        elif isinstance(input_image, torch.Tensor):
            rgb = input_image.squeeze()
        else:
            raise TypeError(f"Unknown input type: {type(input_image) = }")
        input_size = rgb.shape
        assert (
            3 == rgb.dim() and 3 == input_size[0]
        ), f"Wrong input shape {input_size}, expected [rgb, H, W]"

        # Resize image
        if processing_res > 0:
            rgb = resize_max_res(
                rgb,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )

        # Normalize rgb values
        rgb_norm: torch.Tensor = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = rgb_norm.to(self.dtype)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting --------------
        batched_img = rgb_norm[None]            # [1, 3, H, W]
        _, projected = self.single_infer(
            rgb_in=batched_img,
            noise=noise,
            text_prompt=text_prompt,
        )
        torch.cuda.empty_cache()  # clear vram cache for ensembling
        projected = {k:v[0] for k,v in projected.items()}   # Dict of [C_k, H, W]

        # ----------------- Post processing -----------------
        if post_crf and any([proj_name.startswith('semseg') or proj_name.startswith('matseg') for proj_name in projected.keys()]):
            tgt_h, tgt_w = rgb.shape[1] // 2, rgb.shape[2] // 2
            crf_rgb_input = resize(
                rgb if rgb.ndim >= 3 else rgb.unsqueeze(0),
                (tgt_h, tgt_w),
                interpolation=resample_method,
                antialias=True,
            )
            for proj_name, proj in projected.items():
                if proj_name.startswith('semseg') or proj_name.startswith('matseg'):
                    proj = resize(
                        proj if proj.ndim >= 3 else proj.unsqueeze(0),
                        (tgt_h, tgt_w),
                        interpolation=resample_method,
                        antialias=True,
                    )
                    proj_softmax = torch.softmax(proj, dim=0).cpu().numpy()  # n_cls, H, W

                    posterior = apply_crf(np.ascontiguousarray(chw2hwc(crf_rgb_input).numpy()), proj_softmax)
                    projected[proj_name] = torch.from_numpy(posterior).to(proj.device)

        if match_input_res:     
            for proj_name, proj in projected.items():
                # Resize back to original resolution
                projected[proj_name] = resize(
                    proj if proj.ndim >= 3 else proj.unsqueeze(0),
                    (input_size[-2],input_size[-1]),
                    interpolation=resample_method,
                    antialias=True,
                ).squeeze()

        # Task-specific post-processing
        for proj_name, proj in projected.items():
            if proj_name.startswith('depth'):
                proj = (torch.clip(proj, -1.0, 1.0) + 1.0) / 2.0
            elif proj_name.startswith('normal'):
                proj /= (torch.norm(proj, p=2, dim=0, keepdim=True)+1e-5)
            elif proj_name.startswith('boundary'):
                proj = torch.clip(proj, 0.0, 1.0)
            projected[proj_name] = proj
        
        return UnidOutput(
            projected=projected,
        )

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,   # [1, 3, H, W]
        noise="gaussian",
        text_prompt="",
    ) -> torch.Tensor:
        """ Perform an individual prediction without ensembling.
        """
        device = self.device
        rgb_in = rgb_in.to(device)
        B, _, H, W = rgb_in.shape

        # Get input latents (rgb + noise)
        rgb_latent, latent = self.prepare_input_latents(rgb_in, noise, device, self.dtype) # [B, 4, h, w], [B, 4, h, w]

        # Get timestep (single-step inference)
        self.scheduler.set_timesteps(1, device=device)
        timestep = self.scheduler.timesteps[0]

        # Get text embedding
        if self.text_embed is None or text_prompt != self.text_prompt:
            self.encode_text_prompt(text_prompt)    # reset if not set or changed
        batch_text_embed = self.text_embed.repeat((B, 1, 1))  # [B, 2, 1024]
        class_emb = self.get_class_emb(class_id=1, bs=B, device=device)

        # Single UNet forward pass & collect intermediate features
        self._set_store_features(True)
        unet_output, interm_feats = self.single_forward(
            rgb_latent=rgb_latent,
            noisy_latent=latent,
            timesteps=timestep,
            unet_kwargs={"encoder_hidden_states": batch_text_embed, "class_labels": class_emb},
            return_intermediate_feats=True,
        )
        self._set_store_features(False)

        # Compute task-specific latents
        latent_dict = dict()    # task_name: task_latent
        for task_name, connector in self.connectors.items():
            noise_p = connector(interm_feats)
            scheduler_step = self.scheduler.step(noise_p, timestep, latent)
            latent_dict[task_name] = scheduler_step.pred_original_sample
        
        # Decode and project to task-specific output space
        if len(self.cnct_names) > 0:    # went through connectors
            task_latents = torch.cat([latent_dict[k] for k in self.cnct_names], dim=0)  # [num_tasks*B, 4, h, w]
            decoded = self.decode_latent(task_latents)    # [num_tasks*B, D, H, W]
            decoded = {k: decoded[i:i+1] for i,k in enumerate(self.cnct_names)}  # task_name: [B, D, H, W]
            projected = {k: self.projectors[k](decoded[k.split('-')[0]]) for k in self.projectors.keys()}  # proj_name: [B, C_k, H, W]
        else:  # directly decode the UNet output without connectors
            scheduler_step = self.scheduler.step(unet_output.sample, timestep, latent)
            latent = scheduler_step.pred_original_sample 
            decoded = self.decode_latent(latent)    # [B, 3, H, W]
            projected = {k: self.projectors[k](decoded) for k in self.projectors.keys()}  # proj_name: [B, C, H, W]

        return decoded, projected

    def single_forward(self, rgb_latent, noisy_latent, timesteps, unet_kwargs, unet=None, return_intermediate_feats=False) -> (torch.Tensor, Union[None, Dict]):
        """ Perform a single forward pass of the UNet model.
        """
        unet_input = torch.cat([rgb_latent, noisy_latent], dim=1)
        if unet is None:
            unet = self.unet
        output = unet(unet_input, timesteps, **unet_kwargs)

        if return_intermediate_feats:
            interm_feats = self._get_matched_module_attr('TemporalTransformerBlock', 'cached_output')
            dev_bs, _, lat_h, lat_w = rgb_latent.shape
            for k, v in interm_feats.items():
                layer_idx = int(k.split('.')[1]) if k.split('.')[1].isdigit() else None
                if k.startswith('down_blocks'):
                    h, w = math.ceil(lat_h / (2**layer_idx)), math.ceil(lat_w / (2**layer_idx))
                elif k.startswith('up_blocks'):
                    h, w = math.ceil(lat_h / (2**(3-layer_idx))), math.ceil(lat_w / (2**(3-layer_idx)))
                elif k.startswith('mid_block'):
                    h, w = math.ceil(lat_h / 8), math.ceil(lat_w / 8)
                interm_feats[k] = v.reshape(dev_bs, h, w, -1).permute(0, 3, 1, 2)
            interm_feats['output'] = output.sample
            return output, interm_feats
        else:
            return output


    ###### ====== Helper functions ====== ######

    def eval(self):
        self.unet.eval()
        for proj_name in self.projectors.keys():
            self.projectors[proj_name].eval()
    
    def to(self, device):
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        for con_name in self.connectors.keys():
            self.connectors[con_name].to(device)
        for proj_name in self.projectors.keys():
            self.projectors[proj_name].to(device)
        return self
    
    def prepare_input_latents(self, input_rgb, noise_type, device=None, dtype=None):
        """ Prepare RGB latent and noisy latent for the UNet model.
        """
        device = input_rgb.device if device is None else device
        dtype = input_rgb.dtype if dtype is None else dtype
        rgb_latents = self.encode_rgb(input_rgb.to(device, dtype=dtype))

        if (noise_type is None) or (noise_type == "zeros"):
            noisy_latent = torch.zeros_like(rgb_latents).to(device, dtype=dtype)
        elif noise_type == "gaussian":
            noisy_latent = torch.randn_like(rgb_latents).to(device, dtype=dtype)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")
        return rgb_latents, noisy_latent


    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """ Encode image [B, 3, H, W] into latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.vae.config.scaling_factor
        return rgb_latent
    

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """ Decode depth latent into depth map.
        """
        # scale latent
        latent = latent / self.vae.config.scaling_factor

        # decode
        z = self.vae.post_quant_conv(latent)
        decoded = self.vae.decoder(z)

        return decoded
    
    def encode_text_prompt(self, prompt=""):
        """
        Encode text embedding for text prompt
        """
        print(f'Encoding new text prompt: \"{self.text_prompt}\" ==> \"{prompt}\"')
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        self.text_prompt = prompt

    def get_class_emb(self, class_id, bs, device, class_num=2):
        x = [0 for _ in range(class_num)]
        x[class_id] = 1
        class_emb = torch.tensor(x).float().unsqueeze(0).to(device)
        class_emb = torch.cat([torch.sin(class_emb), torch.cos(class_emb)], dim=-1).repeat(bs, 1)
        return class_emb

    def init(self, is_video, is_train, unet=None):
        """ Initialize the UNet model for video processing and training.
        """
        if unet is None:
            unet = self.unet
        for key, attn_processor in unet.attn_processors.items():
            attn_processor.init(is_video=is_video, is_train=is_train)

    def prepare_backward(self):
        """ Set the UNet model to be ready for backward pass (training).
        """
        self.set_all_proc_attr('_is_backward', True)

    def _set_store_features(self, value):
        """ Set the '_store_features' attribute for all TemporalTransformerBlock modules in the UNet model.
            This is used for enabling/disabling the extraction of intermediate features during the forward pass.
        """
        self._set_matched_module_attr('TemporalTransformerBlock', '_store_features', value)

    def set_all_proc_attr(self, attr, value):
        """ Set an attribute for all attention processors in the UNet model.
        """
        for key, attn_processor in self.unet.attn_processors.items():
            setattr(attn_processor, attr, value)

    def _set_matched_module_attr(self, mod_name, attr, value):
        """ Set an attribute for all modules with a specific class name in the UNet model.
        """
        for module in self.unet.modules():
            if module.__class__.__name__ == mod_name:
                setattr(module, attr, value)

    def _get_matched_module_attr(self, mod_name, attr):
        """ Get an attribute for all modules with a specific class name in the UNet model.
        """
        attrs = {}
        for name, module in self.unet.named_modules():
            if module.__class__.__name__ == mod_name:
                attrs[name] = getattr(module, attr)
        return attrs
    
    ###### ====== Model building and loading functions ====== ######
    
    def build_model(self, is_train: bool, inf_params=None, **kwargs):
        """ Build a model 
        """
        if is_train:
            assert inf_params is None, "inf_params should be None during training."
        
        msg = []
        if '(dec)' in self.arch:
            self.dec_dim = self.vae.decoder.conv_out.in_channels
            self.vae.decoder.conv_out = torch.nn.Identity()
            msg.append(f"Arch: {self.arch}  -  Decoder last conv layer removed, Output dim: 3 -> {self.dec_dim}")
        if '(con)' in self.arch:
            con_dim = self.unet.conv_out.in_channels
            self.unet.conv_out = torch.nn.Identity()
            msg.append(f"Arch: {self.arch}  -  UNet last conv layer removed, Output dim: 4 -> {con_dim}")

        self.arch = self.arch.replace('(con)', '').replace('(dec)', '')
        if self.arch == 'MemAttnTemp':
            msg += build_model_MemAttnTemp(self, is_train, inf_params, **kwargs)
        else:
            raise NotImplementedError(f"Architecture {self.arch} not implemented.")
        
        return msg
    
    def build_connector(self, pretrained_path, cnct_name, cnct_conf):
        """ Build and load a connector.
        """
        connector = DPTConnector(**cnct_conf, name=cnct_name)
        connector.from_pretrained(osp.join(pretrained_path, 'connectors'))
        self.connectors[cnct_name] = connector
        output_dim, fusion_hidden_size = cnct_conf['output_dim'], cnct_conf['fusion_hidden_size']
        return f'Arch: {self.arch}  -  DPTConnector {cnct_name}(output_dim={output_dim}, fusion_hidden_size={fusion_hidden_size}) loaded successfully.'
    
    def build_projector(self, pretrained_path, proj_name, hidden_dims, output_dim):
        """ Build and load a projector.
        """
        state_dict = torch.load(osp.join(pretrained_path, 'projectors', f"{proj_name}.bin"), map_location='cpu')
        projector = TaskProjector(
            output_dim=output_dim, hidden_dims=hidden_dims, name=proj_name, input_conv=self.dec_dim,
        )
        projector.from_pretrained(osp.join(pretrained_path, 'projectors'))
        self.projectors[proj_name] = projector
        return f'Arch: {self.arch}  -  TaskProjector {proj_name}(hidden={hidden_dims},out_dim={output_dim}) loaded successfully.'
    
    def remove_projector(self, proj_name):
        """ Remove a projector from the pipeline.
        """
        if proj_name in self.projectors:
            del self.projectors[proj_name]
            self.proj_names.remove(proj_name)
            return True
        return False
    
    def remove_connector(self, cnct_name):
        """ Remove a connector from the pipeline.
        """
        if cnct_name in self.connectors:
            del self.connectors[cnct_name]
            self.cnct_names.remove(cnct_name)
            return True
        return False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        """ Load a pipeline from a pretrained checkpoint.
        """
        base_model_path = kwargs.pop("base_model_path", "checkpoints/pretrained_sd2")
        is_ckpt = '/checkpoint-' in pretrained_model_name_or_path
        arg_path = os.path.join(osp.dirname(pretrained_model_name_or_path), "arguments.txt") if is_ckpt else os.path.join(pretrained_model_name_or_path, "arguments.txt")
        with open(arg_path, "r") as f:
            arguments = [l.strip() for l in f.readlines()]
        
        arg = eval([l for l in arguments if l.startswith("_content:")][0].replace("_content:", "").strip())
        if "architecture" in arg:
            kwargs["architecture"] = arg["architecture"]
        pipe = super().from_pretrained(base_model_path, **kwargs)  # load base model first
        msg = pipe.build_model(is_train=kwargs.get("is_train", False), noise_type=kwargs.get('noise_type', None), inf_params=kwargs.get("inf_params", None), enable_xformers=True)

        # Load all projectors defined during training
        tasks_to_load = arg['task_to_train'] if type(arg['task_to_train']) is list else [arg['task_to_train']]
        for task in tasks_to_load:
            for proj_name, proj_info in arg['all_projectors'][task].items():
                p_msg = pipe.build_projector(pretrained_model_name_or_path, proj_name, hidden_dims=proj_info['hidden_dims'], output_dim=proj_info['proj_dim'])
                msg.append(p_msg)
        
        # Load all connectors defined during training
        dist_arg = arg.get('distill_tasks', None)
        con_arg = arg.get('connectors', None)
        if dist_arg is not None and con_arg is not None:
            for cnct_name in dist_arg.keys():
                c_msg = pipe.build_connector(
                    pretrained_model_name_or_path, 
                    cnct_name,
                    cnct_conf=con_arg['config'],
                )
                msg.append(c_msg)
        pipe.cnct_names = list(pipe.connectors.keys())
        pipe.proj_names = list(pipe.projectors.keys())

        load_state = safetensors.torch.load_file(os.path.join(pretrained_model_name_or_path, 'unet', 'diffusion_pytorch_model.safetensors'))

        if pipe.unet.config['in_channels'] != load_state['conv_in.weight'].shape[1]:
            repeat = pipe.unet.config['in_channels'] // load_state['conv_in.weight'].shape[1]
            replace_unet_conv_in(pipe.unet, repeat=2)
            msg.append(f"Inference modification: Unet conv_in layer(ch=4) is loaded, need to be replaced to (ch=8) for proper loading.")

        pipe.unet.load_state_dict(load_state, strict=True)
        print('Final state_dict loaded strictly.')
        return pipe, msg

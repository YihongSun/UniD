### Adapted from https://github.com/Jeff-LiangF/streamv2v

from importlib import import_module
from typing import Callable, Optional, Union, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class TemporalSTXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.
    Adapted from https://github.com/Jeff-LiangF/streamv2v/blob/3e571d88456f85b97f68f321c74327c338a9c194/src/streamv2v/models/attention_processor.py

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None, name=None, 
                 interval=1, max_frames=16, enable_memory=True, temp_pos_emb=None, pop_loc=1):
        self.attention_op = attention_op
        self.name = name
        self.interval = interval
        self.max_frames = max_frames
        self._enable_memory = enable_memory
        self._temporal_pos_emb = temp_pos_emb
        self._pop_loc = pop_loc

        self.cached_key = list()
        self.cached_value = list()
        self.frame_id = 0
        self._is_train = None
        self._is_backward = None
        self._is_video = None
    
    def init(self, is_video, is_train):
        """ Initialize the processor and resets the memory of the cached keys, values, and outputs.
        """
        self._is_video = is_video
        self._is_train = is_train
        self._is_backward = False

        self.cached_key.clear()
        self.cached_value.clear()
        self.frame_id = 0
    
    def set_inference_param(self, max_frames, interval, enable_memory, pop_loc):
        """ Sets the inference parameters for the attention processor.
        """
        self.max_frames = max_frames
        self.interval = interval
        self._enable_memory = enable_memory
        self._pop_loc = pop_loc

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        """ Forward pass
        """
        B, N, D = hidden_states.shape
        residual = hidden_states    # (B, N, D)

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)         # (B, N, C)

        is_selfattn = False
        if encoder_hidden_states is None:
            is_selfattn = True
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)   # (B, N, C)
        value = attn.to_v(encoder_hidden_states, *args) # (B, N, C)

        bool_use_mem = self._enable_memory and is_selfattn and self._is_video
        if bool_use_mem:
            cached_key = key.clone()
            cached_value = value.clone()   

            # perform a deque operation with backward
            if self._is_backward:  
                self.cached_key = self.cached_key[:-1]
                self.cached_value = self.cached_value[:-1]

            emb_weights = self._temporal_pos_emb.weight  # (8, D)
            curr_emb = emb_weights[0:2]     # (2, D)
            start_emb = emb_weights[2:4]    # (2, D)
            end_emb = emb_weights[4:6]      # (2, D)

            key_embs, val_embs = [], []
            if len(self.cached_key) > 0:
                w = torch.linspace(
                    0, 1, len(self.cached_key), device=key.device
                ).unsqueeze(1)      # (K, 1)
                key_emb_local = start_emb[:1] * (1 - w) + end_emb[:1] * w  # (K, D)
                val_emb_local  = start_emb[1:] * (1 - w) + end_emb[1:] * w  # (K, D)

                key_emb_local = key_emb_local[:, None, None].repeat(1, B, N, 1)  # (K, B, N, D)
                val_emb_local = val_emb_local[:, None, None].repeat(1, B, N, 1)  # (K, B, N, D)

                key_embs += [x for x in key_emb_local]
                val_embs += [x for x in val_emb_local]

            key_embs.append(curr_emb[None, :1, :].repeat(B, N, 1))
            val_embs.append(curr_emb[None, 1:, :].repeat(B, N, 1))

            key = torch.cat(self.cached_key + [key], dim=1) + torch.cat(key_embs, dim=1)
            value = torch.cat(self.cached_value + [value], dim=1) + torch.cat(val_embs, dim=1)

        query = attn.head_to_batch_dim(query).contiguous()      # (B*5, N, C/5)
        key = attn.head_to_batch_dim(key).contiguous()          # (B*5, N, C/5)
        value = attn.head_to_batch_dim(value).contiguous()      # (B*5, N, C/5)
        
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        ## Code for debug
        # if self.name == 'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor':
        # if self.name == 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor':
        #     print("back:", self._is_backward, "is_video:", self._is_video, 'hidden_state_norm:', torch.norm(hidden_states), "len_cache:", len(self.cached_key), "qkv_shapes:", query.shape, key.shape, value.shape)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)   # (B, N, C)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
    
        if bool_use_mem:
            is_infer = not self._is_train
            mem_stride = self.interval if is_infer else 1

            if not self._is_backward and self.frame_id % mem_stride == 0:
                self.cached_key.append(cached_key)
                self.cached_value.append(cached_value)

            if is_infer and len(self.cached_key) > self.max_frames:
                self.cached_key.pop(self._pop_loc)
                self.cached_value.pop(self._pop_loc)

            self.frame_id += 1 
    
        return hidden_states
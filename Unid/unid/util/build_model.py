import torch
import torch.nn as nn
from torch.nn import Conv2d, Parameter
from typing import Optional, Callable, Dict, Any, Type
from packaging import version
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers.utils.import_utils import is_xformers_available
from .attention_processor import TemporalSTXFormersAttnProcessor
from .attention_block import TemporalTransformerBlock

def wrap_modules_with_path_tracking(
    module: nn.Module, 
    target_class: Type[nn.Module], 
    wrapper_class: Type[nn.Module], 
    path: str = ""
) -> None:
    """ Recursively wrap modules while tracking the full path for debugging.
    """
    children = list(module.named_children())
    
    for name, child in children:
        current_path = f"{path}.{name}" if path else name
        
        if isinstance(child, target_class):
            wrapped_module = wrapper_class(child, path=current_path)
            setattr(module, name, wrapped_module)
        else:
            wrap_modules_with_path_tracking(child, target_class, wrapper_class, current_path)

def traverse_modules(
    module: nn.Module,
    path_cond_fn: Callable[[str], bool],
    out_collection: Dict[str, nn.Module] = {},
    path: str = ""
) -> Dict[str, nn.Module]:
    """ Traverse modules and collect those that satisfy a condition, tracking their paths.
    """

    children = list(module.named_children())
    for name, child in children:
        current_path = f"{path}.{name}" if path else name

        if path_cond_fn(current_path):
            out_collection[current_path] = child
        else:
            traverse_modules(child, path_cond_fn, out_collection, current_path)
    return out_collection

def build_model_MemAttnTemp(pipeline, is_train, inf_params, **kwargs):
    arch_name = pipeline.arch
    processor_cls = {
        'MemAttnTemp': TemporalSTXFormersAttnProcessor,
    }[arch_name]
    unet = pipeline.unet
    msg = []
    
    ### modify the UNet structure that is configurable (change is automatically propagated when loading).
    if is_train and kwargs.get('noise_type', None) is not None:
        assert unet.config['in_channels'] == 4, "original input channels should be 4."
        replace_unet_conv_in(unet, repeat=2)
        msg.append(f"Arch: {arch_name}  -  Unet conv_in layer(ch=4) is replaced to (ch=8) for RGB-depth or RGB-normals input.")

    ### modify the UNet structure that is not configurable (change is not automatically propagated when loading - needs manual call).
    wrap_modules_with_path_tracking(
        unet, target_class=BasicTransformerBlock, wrapper_class=TemporalTransformerBlock
    )
    msg.append(f"Arch: {arch_name}  -  BasicTransformerBlock is replaced to TemporalTransformerBlock.")

    if kwargs.get('enable_xformers', False):
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                msg.append(
                    "WARNING: xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()
            msg.append(f"Arch: {arch_name}  -  xFormers is enabled for memory efficient attention.")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ### modify the attention processors
    attn_processors = unet.attn_processors
    new_attn_processors = {}
    for key, attn_processor in attn_processors.items():
        block_path = '.'.join(key.split('.')[:-2])
        block = traverse_modules(unet, path_cond_fn=lambda s: s==block_path)[block_path]

        assert isinstance(attn_processor, XFormersAttnProcessor), \
                f"We only replace 'XFormersAttnProcessor' to '{processor_cls.__name__}'"
        new_attn_processors[key] = processor_cls(
            name=key, temp_pos_emb=block.temp_pos_emb
        )
    unet.set_attn_processor(new_attn_processors)
    msg.append(f"Arch: {arch_name}  -  Swapped in {processor_cls.__name__}.")

    ### modify the inference parameters
    if not is_train:
        max_frames = inf_params.get("len", 16)
        interval = inf_params.get("st", 1)
        enable_memory = True
        pop_loc = inf_params.get("pop_loc", 0)

        for key, attn_processor in unet.attn_processors.items():
            assert isinstance(attn_processor, processor_cls), \
                    f"Processor must be '{processor_cls.__name__}' during inference."
            unet.attn_processors[key].set_inference_param(
                max_frames=max_frames, interval=interval, enable_memory=enable_memory, pop_loc=pop_loc
            )
        msg.append(f"Arch: {arch_name}  -  Swapped in {processor_cls.__name__}(len={max_frames},st={interval},enable_memory={enable_memory},pop_loc={pop_loc}).")

    return msg

def replace_unet_conv_in(unet, repeat=2):
    _weight = unet.conv_in.weight.clone() 
    _bias = unet.conv_in.bias.clone() 
    _weight = _weight.repeat((1, repeat, 1, 1))  
    # scale the activation magnitude
    _weight /= repeat
    _bias /= repeat
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4*repeat, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    # replace config
    unet.config['in_channels'] = 4*repeat
    return
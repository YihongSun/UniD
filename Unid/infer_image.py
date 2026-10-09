import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.multiprocessing as mp
from torch.utils.data import Subset

# add
import sys
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
sys.path.append(os.getcwd())

from src.util.seed_all import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)

"""
This script runs single-image dense prediction across multiple GPUs by splitting the dataset and launching separate processes for each GPU.
"""

def load_model(checkpoint_path, model_type, timestep_spacing, proj_name, dtype, variant, device):
    """Load the model and return the pipeline."""
    print(f"Loading model from {checkpoint_path} on device {device}")
    
    if model_type == "unid":
        from unid import UnidPipeline
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        unet         = UNet2DConditionModel.from_pretrained('checkpoints/pretrained_sd2', subfolder="unet")   
        vae          = AutoencoderKL.from_pretrained('checkpoints/pretrained_sd2', subfolder="vae")  
        text_encoder = CLIPTextModel.from_pretrained('checkpoints/pretrained_sd2', subfolder="text_encoder")  
        tokenizer    = CLIPTokenizer.from_pretrained('checkpoints/pretrained_sd2', subfolder="tokenizer") 
        scheduler    = DDIMScheduler.from_pretrained('checkpoints/pretrained_sd2', timestep_spacing=timestep_spacing, subfolder="scheduler") 
        pipe, msg = UnidPipeline.from_pretrained(
            pretrained_model_name_or_path = checkpoint_path,
            unet=unet, 
            vae=vae, 
            scheduler=scheduler, 
            text_encoder=text_encoder, 
            tokenizer=tokenizer, 
            variant=variant, 
            torch_dtype=dtype,
            inf_params={'len': 0, 'st': 1, 'pop_loc': 0}
        )
        [logging.info(m) for m in msg]
    elif model_type == "dino":
        from baselines.dino_pipeline import DinoPipeline
        pipe, msg = DinoPipeline.from_pretrained(
            checkpoint_path,
            device=device,
            dtype=dtype
        )
        [logging.info(m) for m in msg]
    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")
    
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    pipe = pipe.to(device)
    pipe.unet.eval()
    pipe.eval()
    
    return pipe


def worker_process(gpu_id, indices, args, dataset_config, base_data_dir):
    """Worker process that runs inference on a subset of images on a specific GPU."""
    
    # Set device for this worker
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Set random seed (different for each GPU to ensure different ensemble samples if needed)
    seed = args.seed + gpu_id if args.seed is not None else None
    if seed is not None:
        seed_all(seed)
    
    # Setup logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {gpu_id}] %(levelname)s - %(message)s'
    )
    
    # Determine dtype and variant
    if args.half_precision:
        dtype = torch.float16
        variant = "fp16"
    else:
        dtype = torch.float32
        variant = None
    
    # Load model on this GPU
    pipe = load_model(
        args.checkpoint,
        args.model_type,
        args.timestep_spacing,
        args.proj,
        dtype,
        variant,
        device
    )
    
    # Setup projectors
    proj_names = list(pipe.projectors.keys())
    for proj_name in proj_names:
        if proj_name != args.proj:
            assert pipe.remove_projector(proj_name)
    conn_names = list(pipe.connectors.keys())
    for conn_name in conn_names:
        if not args.proj.startswith(conn_name):
            assert pipe.remove_connector(conn_name)
    
    # Load dataset
    cfg_data = OmegaConf.load(dataset_config)
    dataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )
    
    # Create subset for this worker
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=1, num_workers=0)
    
    logging.info(f"GPU {gpu_id}: Processing {len(indices)} images")

    if args.label_mapping is not None:
        assert args.save_argmax, "Label mapping can only be applied when --save_argmax is enabled."
        label_map = np.load(args.label_mapping)
        ignore_map = (label_map == -1)
    else:
        label_map, ignore_map = None, None

    # Run inference
    with torch.no_grad():
        for batch in tqdm(
            dataloader, 
            desc=f"GPU {gpu_id}", 
            position=gpu_id,
            leave=True
        ):
            # Read input image
            rgb_int = batch["rgb_int"].squeeze().numpy().astype(np.uint8)
            rgb_int = np.moveaxis(rgb_int, 0, -1)
            input_image = Image.fromarray(rgb_int)
            
            # Run inference based on model type
            if args.model_type == "unid":
                pipe_out = pipe(
                    input_image,
                    processing_res=args.processing_res,
                    match_input_res=not args.output_processing_res,
                    resample_method=args.resample_method,
                    noise=args.noise,
                    text_prompt=args.text_prompt,
                )
            elif args.model_type == "dino":
                pipe_out = pipe(
                    input_image,
                    processing_res=args.processing_res,
                    match_input_res=not args.output_processing_res,
                )
            
            pred = pipe_out.projected[args.proj].cpu().numpy()
            
            # Save predictions
            rgb_filename = batch["rgb_relative_path"][0]
            rgb_basename = os.path.basename(rgb_filename)
            scene_dir = os.path.join(args.output_dir, os.path.dirname(rgb_filename))
            os.makedirs(scene_dir, exist_ok=True)
            
            pred_basename = get_pred_name(
                rgb_basename, dataset.name_mode, suffix=".npy"
            )
            save_to = os.path.join(scene_dir, pred_basename)
            
            if args.save_png:
                img = np.clip(pred.transpose((1, 2, 0)), -1, 1) / 2 + 0.5
                img = (img * 255).astype(np.uint8)
                png_save_to = save_to.rstrip('.npy') + '.png'
                Image.fromarray(img).save(png_save_to)
            elif args.save_argmax:

                if ignore_map is not None:
                    pred[ignore_map] = -1e9

                img = np.argmax(pred, axis=0)

                if label_map is not None:
                    img = label_map[img]

                if img.max() < 256:
                    img = img.astype(np.uint8)
                elif img.max() < 65536:
                    img = img.astype(np.uint16)
                else:
                    raise ValueError("Argmax image has too many classes to be saved as PNG.")
                png_save_to = save_to.rstrip('.npy') + '.png'
                Image.fromarray(img).save(png_save_to)
            else:
                np.save(save_to, pred)
    
    logging.info(f"GPU {gpu_id}: Completed processing {len(indices)} images")


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/marigold-e2e-ft-depth",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    # add
    parser.add_argument(
        "--noise",
        type=str,
        default='zeros',
        choices=["gaussian", "pyramid", "zeros"],
    )
    parser.add_argument(
        "--timestep_spacing",
        type=str,
        default='trailing',
        choices=["trailing", "leading"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='unid',
        choices=["unid", "dino"],
    ) 

    # unified model
    parser.add_argument(
        "--text_prompt",
        type=str,
        default='',
    )
    parser.add_argument(
        "--proj",
        type=str,
        default="depth",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Also save the predicted depth as PNG images.",
    )
    parser.add_argument(
        "--save_argmax",
        action="store_true",
        help="Also save the predicted depth as PNG images.",
    )
    parser.add_argument(
        "--label_mapping",
        type=str,
        default=None,
    )  

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    
    # Parallelization argument
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use. If None, uses all available GPUs.",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.base_data_dir):
        parser.error(
            f"dataset root does not exist: {args.base_data_dir}. Obtain the dataset locally and pass its path with --base_data_dir."
        )
    if not os.path.isfile(args.dataset_config):
        parser.error(f"dataset config does not exist: {args.dataset_config}")

    print(f"arguments: {args}")
    
    # Save arguments
    parent_output_dir = os.path.dirname(args.output_dir)
    os.makedirs(parent_output_dir, exist_ok=True)
    args_dict = vars(args)
    args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
    args_path = os.path.join(parent_output_dir, "arguments.txt")
    with open(args_path, 'w') as file:
        file.write(args_str)
    print(f"Arguments saved in {args_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of GPUs
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available == 0:
        logging.error("No GPUs available. This script requires CUDA.")
        exit(1)
    
    num_gpus = args.num_gpus if args.num_gpus is not None else num_gpus_available
    num_gpus = min(num_gpus, num_gpus_available)
    
    logging.info(f"Using {num_gpus} out of {num_gpus_available} available GPUs")
    
    # Set random seed for dataset split
    if args.seed is None:
        import time
        args.seed = int(time.time())
    
    # Load dataset to get total number of images
    cfg_data = OmegaConf.load(args.dataset_config)
    dataset = get_dataset(
        cfg_data, base_data_dir=args.base_data_dir, mode=DatasetMode.RGB_ONLY
    )
    total_images = len(dataset)
    
    logging.info(f"Total images: {total_images}")
    logging.info(f"Images per GPU: ~{total_images // num_gpus}")
    
    # Split indices across GPUs
    all_indices = list(range(total_images))
    indices_per_gpu = [
        all_indices[i::num_gpus] for i in range(num_gpus)
    ]
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Launch worker processes
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, indices_per_gpu[gpu_id], args, args.dataset_config, args.base_data_dir)
        )
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    logging.info("All GPUs completed processing!")

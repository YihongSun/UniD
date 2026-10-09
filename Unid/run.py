import argparse, logging, os, time, random
import os.path as osp
from glob import glob
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

from unid import UnidPipeline
import sys; sys.path.append('./')
from unid.util.image_util import chw2hwc, colorize_depth_maps
from unid.util.projector_selection import select_projectors

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def seed_all(seed: int = 0):
    """ Set random seeds of all components.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = argparse.ArgumentParser(description="Run single-image depth estimation using Unid."
    )
    # Required settings
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Local UniD checkpoint directory."
    )
    parser.add_argument(
        "--base_model", type=str, default="checkpoints/pretrained_sd2",
        help="Local Diffusers-format Stable Diffusion 2 base model directory (default: checkpoints/pretrained_sd2).",
    )
    parser.add_argument(
        "--input_rgb", type=str, required=True, help="Path to one image or a directory of image frames."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # Infernece settings
    parser.add_argument(
        "--projs", type=str, nargs="+", default=None   # load all proj if None
    )
    parser.add_argument(
        "--model_params",
        type=str,
        default='(len=16,st=1,pop_loc=0)',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='unid',
        choices=["unid", "dino"],
    ) 
    parser.add_argument(
        "--post_crf",
        action="store_true",
        default=False,
        help="Whether to apply CRF post-processing for segmentation maps. Default: False.",
    )
    parser.add_argument(
        "--text_prompt", type=str, default="video_dense_pixel_prediction"
    )
    parser.add_argument(
        "--processing_res", type=int, default=768, help="Maximum resolution of processing. 0 for using input image resolution. Default: 768."
    )
    parser.add_argument(
        "--output_processing_res", action="store_true", help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )

    # other settings
    parser.add_argument(
        "--seed", type=int, default=42, help="Reproducibility seed. Set to `None` for unseeded inference."
    )
    parser.add_argument(
        "--noise", type=str, default='zeros', choices=["gaussian", "pyramid", "zeros"]
    )
    parser.add_argument(
        "--timestep_spacing", type=str, default='trailing', choices=["trailing", "leading"]
    )
    parser.add_argument(
        "--resample_method", choices=["bilinear", "bicubic", "nearest"], default="bilinear", help="Resampling method used to resize images and depth predictions."
    )
    parser.add_argument(
        "--half_precision", "--fp16", action="store_true", help="Run with half-precision (16-bit float), might lead to suboptimal result."
    )

    # Saving settings
    parser.add_argument(
        "--depth_color_map", type=str, default="Spectral", help="Colormap used to render depth predictions."
    )
    parser.add_argument(
        "--store_np", action="store_true", default=False, help="Save prediction as np array."
    )

    # Video input settings
    parser.add_argument(
        "--stride", type=int, default=None, help="Stride of input temporal frames."
    )
    parser.add_argument(
        "--early_break", type=int, default=None, help="Early stop of temporal rollouts.",
    )
    return parser

def get_args():
    """ Get command line arguments.
    """
    args = get_parser().parse_args()
    args.model_params = eval(f'dict{args.model_params}')    # convert string of dict to dict, e.g. "({'enc_st': 3, 'post_thrd': 0.5})" -> {'enc_st': 3, 'post_thrd': 0.5}
    return args

# python Unid/run.py --checkpoint=<CKPT> --input_rgb=<INPUT_DIR> --output_dir=<OUTPUT_DIR>
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    # -------------------- Settings --------------------
    checkpoint_path = args.checkpoint
    base_model_path = args.base_model
    input_rgb = args.input_rgb
    output_dir = args.output_dir
    match_input_res = not args.output_processing_res

    seed = args.seed if args.seed is not None else int(time.time())
    seed_all(seed)

    # -------------------- Preparation --------------------
    logging.info(
        f"Run settings: checkpoint = `{checkpoint_path}`, base_model = `{base_model_path}`, "
        f"input_rgb = `{input_rgb}`, output_dir = `{output_dir}`, "
        f"model_params = {args.model_params}, post_crf = {args.post_crf}, text_prompt = `{args.text_prompt}`, processing_res = {args.processing_res}"
    )
    
    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError(
            "UniD inference requires CUDA and xFormers memory-efficient attention. "
            "No CUDA device is available; see README.md for the tested GPU setup."
        )
    logging.info(f"device = {device}")

    # -------------------- Input --------------------
    if not os.path.exists(input_rgb):
        raise FileNotFoundError(f"Input path does not exist: {input_rgb}")
    if os.path.splitext(input_rgb)[1].lower() in EXTENSION_LIST:
        logging.info(f"Detected input as a single image file.")
        rgb_filename_list = [input_rgb]
    else:
        logging.info(f"Detected input as a dir of image files.")
        rgb_filename_list = glob(os.path.join(input_rgb, "*"))
        rgb_filename_list = [
            f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        rgb_filename_list = sorted(rgb_filename_list)

    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb}'")
        exit(1)
    if args.stride is not None and args.stride > 1:
        org_len = len(rgb_filename_list)
        rgb_filename_list = rgb_filename_list[::args.stride]
        logging.info(f"Temporal stride applied, only running on every {args.stride}-th image, {org_len} -> {len(rgb_filename_list)} images.")
    if args.early_break is not None and 0 < args.early_break < n_images:
        rgb_filename_list = rgb_filename_list[:args.early_break]
        logging.info(f"Early break applied, only running on first {args.early_break} images.")

    # -------------------- Model --------------------
    required_checkpoint_files = [
        "arguments.txt",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
    ]
    required_base_files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "scheduler/scheduler_config.json",
    ]
    for root, required, label in (
        (checkpoint_path, required_checkpoint_files, "UniD checkpoint"),
        (base_model_path, required_base_files, "Stable Diffusion 2 base model"),
    ):
        missing = [rel for rel in required if not os.path.isfile(os.path.join(root, rel))]
        if missing:
            raise FileNotFoundError(
                f"{label} at '{root}' is missing required files: {', '.join(missing)}. "
                "Use a local checkpoint directory with the layout documented in README.md."
            )
    projector_dir = os.path.join(checkpoint_path, "projectors")
    if not os.path.isdir(projector_dir) or not glob(os.path.join(projector_dir, "*.bin")):
        raise FileNotFoundError(
            f"UniD checkpoint at '{checkpoint_path}' has no projector weight files under projectors/. "
            "Use a compatible local checkpoint; see README.md for its layout."
        )

    if args.half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = torch.float32
        variant = None

    logging.info(f"Loading Model: {checkpoint_path}")
    unet         = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")
    vae          = AutoencoderKL.from_pretrained(base_model_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(base_model_path, subfolder="text_encoder")
    tokenizer    = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
    scheduler    = DDIMScheduler.from_pretrained(base_model_path, timestep_spacing=args.timestep_spacing, subfolder="scheduler")
    pipe, msg = UnidPipeline.from_pretrained(
        pretrained_model_name_or_path = checkpoint_path,
        base_model_path=base_model_path,
        unet=unet, 
        vae=vae, 
        scheduler=scheduler, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        variant=variant, 
        torch_dtype=dtype,
        inf_params=args.model_params
    )
    [logging.info(m) for m in msg]
    
    pipe = pipe.to(device)
    pipe.init(is_video=True, is_train=False)
    pipe.eval()

    # -------------------- Projectors  --------------------
    if args.projs is not None:
        for proj_name in pipe.projectors.keys():
            if proj_name not in args.projs:
                logging.info(f"Removed projector {proj_name} for inference.")
        for conn_name in pipe.connectors.keys():
            if not np.any([proj_name.startswith(conn_name) for proj_name in args.projs]):
                logging.info(f"Removed connector {conn_name} for inference.")
    proj_names = select_projectors(pipe, args.projs)

    # -------------------- Color codes  --------------------
    proj_color_codes = {}
    if 'semseg-vipseg' in proj_names:
        from dataloaders.datasets.video_datasets.vipseg import VideoVIPSegDataset
        proj_color_codes['semseg-vipseg'] = VideoVIPSegDataset(T=1,full=False).sem_color_code.to(device=device)     # 125, 3
    if 'semseg-cityscapes' in proj_names:
        from dataloaders.datasets.image_datasets.cityscapes import ImageCityscapesDataset
        proj_color_codes['semseg-cityscapes'] = ImageCityscapesDataset(full=False).sem_color_code.to(device=device) # 35, 3
    if 'semseg-coco' in proj_names:
        from dataloaders.datasets.image_datasets.coco import ImageCOCODataset
        proj_color_codes['semseg-coco'] = ImageCOCODataset(full=False).sem_color_code.to(device=device)             # 201, 3
    if 'keypoint-coco' in proj_names:
        proj_color_codes['keypoint-coco'] = torch.rand(3,17).to(device=device)
    if 'matseg' in proj_names:
        from dataloaders.datasets.image_datasets.dms import ImageDMSDataset
        proj_color_codes['matseg'] = ImageDMSDataset(full=False).mat_color_code.to(device=device)             # 23, 3
    if 'partseg-coco' in proj_names:
        torch.manual_seed(42)
        code = torch.rand(16,3).to(device=device)
        code[0,:] = 0.0   # background as black
        code[15,:] = 1.0   # placeholder for unknown parts as white
        proj_color_codes['partseg-coco'] = code

    # -------------------- Output Dir  --------------------
    os.makedirs(output_dir, exist_ok=True)
    for proj_name in proj_names:
        os.makedirs(os.path.join(output_dir, proj_name), exist_ok=True)
        if proj_name.startswith('depth') or args.store_np:
            os.makedirs(os.path.join(output_dir, f"_{proj_name}_npy"), exist_ok=True)

    # -------------------- Inference and saving --------------------
    vid_min_max = {"depth": (np.inf, -np.inf)}
    with torch.no_grad():
        for i, rgb_path in enumerate(tqdm(rgb_filename_list, desc=f"Unid Inference", leave=True)):
            # Read input image
            input_image = Image.open(rgb_path)
            pipe_out = pipe(
                input_image,
                processing_res=args.processing_res,
                match_input_res=match_input_res,
                resample_method=args.resample_method,
                noise=args.noise,
                text_prompt=args.text_prompt,
                post_crf=args.post_crf,
            )

            projected_np, projected_vis = dict(), dict()
            for proj_name, proj in pipe_out.projected.items():
                if proj_name.startswith('depth'):
                    proj = proj.cpu().numpy()
                    vid_min_max["depth"] = (
                        min(vid_min_max["depth"][0], proj.min()),
                        max(vid_min_max["depth"][1], proj.max()),
                    )
                    projected_np[proj_name] = proj
                    projected_vis[proj_name] = None # need to go through entire video for min-max
                elif proj_name.startswith('normal') or proj_name.startswith('albedo') or proj_name.startswith('illum') or proj_name.startswith('restore'):
                    proj = proj.cpu().numpy()
                    proj = proj.clip(-1.0, 1.0)
                    colored = (((proj+1)/2) * 255).astype(np.uint8)
                    projected_np[proj_name] = proj
                    projected_vis[proj_name] = Image.fromarray(chw2hwc(colored))
                elif proj_name.startswith('boundary'):
                    proj = proj.cpu().numpy()
                    proj = proj.clip(0.0, 1.0)
                    colored = (proj * 255).astype(np.uint8)
                    projected_np[proj_name] = proj
                    projected_vis[proj_name] = Image.fromarray(np.stack([colored]*3, axis=2))
                elif proj_name.startswith('semseg') or proj_name.startswith('matseg') or proj_name.startswith('partseg'):
                    color_code = proj_color_codes[proj_name]
                    proj_class = torch.argmax(proj, dim=0)  # H, W
                    colored = (color_code[proj_class].cpu().numpy() * 255).astype(np.uint8)
                    projected_np[proj_name] = proj_class
                    projected_vis[proj_name] = Image.fromarray(colored)
                elif proj_name.startswith('keypoint'):
                    color_code = proj_color_codes[proj_name]
                    colored = color_code @ proj.reshape(color_code.shape[-1], -1)  # 3, H*W
                    colored = colored.reshape(3, proj.shape[1], proj.shape[2]).permute(1,2,0).cpu().numpy()
                    projected_np[proj_name] = proj.cpu().numpy()
                    projected_vis[proj_name] = Image.fromarray((colored * 255).astype(np.uint8))
                else:
                    raise NotImplementedError(f"Unknown projector {proj_name}.")

            # Save prediction as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"

            for proj_name in proj_names:
                if projected_vis[proj_name] is not None:
                    out_color_path = os.path.join(output_dir, proj_name, f"{pred_name_base}_colored.png")
                    projected_vis[proj_name].save(out_color_path)
                if proj_name.startswith('depth') or args.store_np:
                    npy_save_path = os.path.join(output_dir, f"_{proj_name}_npy",  f"{pred_name_base}.npy")
                    np.save(npy_save_path, projected_np[proj_name])

        req_post_process = False
        for pn in proj_names:
            if any([pn.startswith(mod) for mod in vid_min_max.keys()]):
                req_post_process = True
                break
        if args.store_np or not req_post_process:
            logging.info(f"Finished estimation, saved to {output_dir}.")
            exit(0)
        
        # Only need to recolorize predictions for depth modality now that we have min-max over the video
        for i, rgb_path in enumerate(tqdm(rgb_filename_list, desc=f"Unid Inference - Video Depth", leave=True)):
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir, "_depth_npy", f"{pred_name_base}.npy")
            pred = np.load(npy_save_path)

            # Normalize depth to [0, 1] for colorization
            min_val, max_val = vid_min_max["depth"]
            if max_val == min_val:
                pred = np.zeros_like(pred)
            else:
                pred = (pred - min_val) / (max_val - min_val)

            # Save colored prediction
            pred = pred.clip(0, 1)
            colored = colorize_depth_maps(
                pred, 0, 1, cmap=args.depth_color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            colored = (colored * 255).astype(np.uint8)
            colored_hwc = chw2hwc(colored)
            pred_colored = Image.fromarray(colored_hwc)
            colored_save_path = os.path.join(output_dir, "depth", f"{pred_name_base}_colored.png")
            pred_colored.save(colored_save_path)

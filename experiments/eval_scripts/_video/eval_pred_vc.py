import os
import os.path as osp
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim

def find_diff(v1, v2, modality):
    """
    Compute angle between two unit vectors (in radians)
    v1, v2: tensors of shape [..., 3]
    """
    if modality == "normal":
        v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
        v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
        angle = torch.acos(torch.clamp(torch.sum(v1 * v2, dim=-1), -1.0, 1.0))
        thrd = 0.19634954084936207/2  # 11.25 degrees in radians
        same = angle < thrd
        return same.numpy(), angle.numpy()
    elif modality in ["albedo", "illum"]:
        thrd = 0.1  # Threshold for considering two values as "same" (can be tuned)
        diff = torch.abs(v1 - v2).mean(dim=-1)
        same = diff < thrd
        return same.numpy(), diff.numpy()

def load_gt(gt_paths, modality):
    if modality == "normal":
        gt_frames = [np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0 * 2 - 1 for p in gt_paths]
        valid_frames = [np.linalg.norm((n+1)/2, axis=-1) > 0 for n in gt_frames]
    elif modality in ["albedo", "illum"]:
        gt_frames = [np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0 for p in gt_paths]
        h, w = gt_frames[0].shape[0:2]
        valid_frames = [np.ones((h, w)) > 0 for n in gt_frames]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return gt_frames, valid_frames

def load_pred(pred_paths, modality, height, width):
    if modality == "normal":
        pred_frames = [np.array(Image.open(p).resize((width, height), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0 * 2 - 1 for p in pred_paths]
    elif modality in ["albedo", "illum"]:
        pred_frames = [np.array(Image.open(p).resize((width, height), Image.Resampling.LANCZOS)).astype(np.float32) / 255.0 for p in pred_paths]
    else:
        raise ValueError(f"Unsupported modality: {modality}")
    return pred_frames

def run_one_scene(arg):
    paths, scene, args = arg
    gt_paths = paths[scene]["gt"]
    pred_paths = paths[scene]["pred"]
    modality = args.modality
    vc_len = args.vc_len

    gt_frames, valid_frames = load_gt(gt_paths, modality)
    pred_frames = load_pred(pred_paths, modality, *gt_frames[0].shape[0:2])
    
    vc_scores, frac_const = [], []
    for start_idx in range(0, len(gt_frames) - vc_len + 1, vc_len):
        vc_valid, vc_pred = None, None
        for i in range(vc_len - 1):
            idx = start_idx + i
            gt_const, gt_diff = find_diff(torch.from_numpy(gt_frames[idx]), torch.from_numpy(gt_frames[idx+1]), modality)
            pred_const, pred_diff = find_diff(torch.from_numpy(pred_frames[idx]), torch.from_numpy(pred_frames[idx+1]), modality)
            valid = valid_frames[idx] & valid_frames[idx+1] & gt_const

            vc_valid = np.logical_and(vc_valid, valid) if vc_valid is not None else valid
            vc_pred = np.logical_and(vc_pred, pred_const) if vc_pred is not None else pred_const

        if vc_valid.sum() == 0:
            continue
        vc_score = vc_pred[vc_valid].mean()
        vc_scores.append(vc_score)
        frac_const.append(vc_valid.sum() / gt_const.size)

        # import imageio
        # imageio.imwrite(f"{scene}_vc_valid.png", (vc_valid.astype(np.uint8)*255))
        # imageio.imwrite(f"{scene}_vc_pred.png", (vc_pred.astype(np.uint8)*255))
        # breakpoint()

    avg_vc = np.mean(vc_scores)
    avg_const = np.mean(frac_const)
    # print(f"Scene {scene} ({avg_const:.2f}) - mean VC-{args.vc_len} : {avg_vc:.4f}")
    return scene, avg_vc

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Prediction VC')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing prediction')
    parser.add_argument('--modality', type=str, required=True, choices=['normal', 'albedo', 'illum'], help='Modality to evaluate')
    parser.add_argument('--vc_len', type=int, default=4, help='Number of consecutive frames to consider for video consistency (default: 4)')
    parser.add_argument('--data_dir', type=str, default='data/interiornet', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./interiornet_vc', help='Directory to save results')
    parser.add_argument('--max_length', type=int, default=45, help='Maximum number of frames to evaluate per video')
    parser.add_argument('--stride', type=int, default=3, help='Stride for frame sampling')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker processes for parallel evaluation')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    data_root = getattr(args, "base_data_dir", getattr(args, "data_dir", None))
    if data_root and not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root is missing: {data_root}. Obtain the dataset from its upstream source "
            "and pass its local path with --base_data_dir or --data_dir."
        )
    pred_dir = args.pred_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    modality = args.modality
    
    assert "interiornet" in data_dir, "This evaluation script is designed for InteriorNet dataset structure. Please provide the correct data_dir."
    folder_name = {
        "normal": "normal0",
        "albedo": "albedo0",
        "illum": "illumination0"
    }[modality]
    os.makedirs(output_dir, exist_ok=True)

    scenes = [f for f in sorted(os.listdir(data_dir)) if osp.isdir(osp.join(data_dir, f))]
    print(f"==> found {len(scenes)} scenes: {scenes[:2]}...")

    paths = {scene: dict() for scene in scenes}
    for scene in scenes:
        gt_scene_dir = osp.join(data_dir, scene, folder_name)
        gt_fnames = sorted([f for f in os.listdir(gt_scene_dir) if f.endswith(".png")])
        if args.stride > 1:
            gt_fnames = gt_fnames[::args.stride]
        if args.max_length > 0 and len(gt_fnames) > args.max_length:
            gt_fnames = gt_fnames[:args.max_length]
        paths[scene]["gt"] = [osp.join(gt_scene_dir, f) for f in gt_fnames]
        paths[scene]["pred1"] = [osp.join(pred_dir, scene, f) for f in gt_fnames]
        paths[scene]["pred2"] = [osp.join(pred_dir, scene, modality, "{}_pred_colored{}".format(*osp.splitext(f))) for f in gt_fnames]
        assert all(os.path.exists(p) for p in paths[scene]["gt"]), f"Not all GT files exist for scene {scene}"
        if all(os.path.exists(p) for p in paths[scene]["pred1"]):
            paths[scene]["pred"] = paths[scene]["pred1"]
        elif all(os.path.exists(p) for p in paths[scene]["pred2"]):
            paths[scene]["pred"] = paths[scene]["pred2"]
        else:
            raise FileNotFoundError(f"Predicted files not found for scene {scene} in either pred1 or pred2 paths")
    
    # run_one_scene((paths, scenes[0], args))  # Test on the first scene to ensure everything works
    with Pool(processes=min(args.num_workers, len(scenes))) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                run_one_scene, 
                [(paths, scene, args) for scene in scenes]
            ),
            total=len(scenes),
            desc="Processing scenes"
        ))
    vc_scenes = dict(results)
    
    with open(osp.join(output_dir, f'{modality}_vc{args.vc_len}.txt'), 'w') as f:
        vc = np.mean(list(vc_scenes.values()))
        f.write(f"Overall Average VC-{args.vc_len} across all scenes: {vc*100:.4f}\n")
        f.write("===================\n")
        f.write("scene_id,vc\n")
        for scene in scenes:
            f.write(f"{scene}: {vc_scenes[scene]}\n")
        print(f"Overall Average VC-{args.vc_len} across all scenes: {vc*100:.4f}")

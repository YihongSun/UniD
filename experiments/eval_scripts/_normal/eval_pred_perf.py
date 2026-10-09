import os
import os.path as osp
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim

def compute_normal_error(pred_norm, gt_norm):
    """ compute per-pixel surface normal error in degrees
        NOTE: pred_norm and gt_norm should be torch tensors of shape (B, 3, ...)
    """
    pred_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
    pred_error = torch.clamp(pred_error, min=-1.0, max=1.0)
    pred_error = torch.acos(pred_error) * 180.0 / np.pi
    pred_error = pred_error.unsqueeze(1)    # (B, 1, ...)
    return pred_error


def compute_normal_metrics(total_normal_errors):
    """ compute surface normal metrics (used for benchmarking)
        NOTE: total_normal_errors should be a 1D torch tensor of errors in degrees
    """
    total_normal_errors = total_normal_errors.detach().cpu().numpy()
    num_pixels = total_normal_errors.shape[0]

    metrics = {
        'mean': np.average(total_normal_errors),
        'median': np.median(total_normal_errors),
        'rmse': np.sqrt(np.sum(total_normal_errors * total_normal_errors) / num_pixels),
        'a1': 100.0 * (np.sum(total_normal_errors < 5) / num_pixels),
        'a2': 100.0 * (np.sum(total_normal_errors < 7.5) / num_pixels),
        'a3': 100.0 * (np.sum(total_normal_errors < 11.25) / num_pixels),
        'a4': 100.0 * (np.sum(total_normal_errors < 22.5) / num_pixels),
        'a5': 100.0 * (np.sum(total_normal_errors < 30) / num_pixels)
    }
    return metrics

def compute_perf(pred_frame, gt_frame, valid_frame, modality):
    if modality == "normal":
        pred = torch.from_numpy(pred_frame).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        gt = torch.from_numpy(gt_frame).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        valid = torch.from_numpy(valid_frame).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        pred = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
        gt = gt / (torch.norm(gt, dim=1, keepdim=True) + 1e-8)
        normal_error = compute_normal_error(pred, gt)  # (1, 1, H, W)
        normal_metrics = compute_normal_metrics(normal_error[valid].view(-1))
        return normal_metrics
    elif modality in ["albedo", "illum"]:
        raise NotImplementedError("Performance metrics for albedo and illumination are not implemented yet.")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

def load_gt(gt_paths, modality):
    if modality == "normal":
        gt_frames = [np.array(Image.open(p).convert('RGB')).astype(np.float32) / 255.0 * 2 - 1 for p in gt_paths]
        assert 'interiornet' in gt_paths[0]
        for f in gt_frames:
            f[:,:,0] = -f[:, :, 0]  # Flip x-axis for normal maps in InteriorNet
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

    gt_frames, valid_frames = load_gt(gt_paths, modality)
    pred_frames = load_pred(pred_paths, modality, *gt_frames[0].shape[0:2])
    
    perf_scores = []
    for pred_frame, gt_frame, valid_frame in zip(pred_frames, gt_frames, valid_frames):
        scores = compute_perf(pred_frame, gt_frame, valid_frame, modality)
        perf_scores.append(scores)
        # import imageio
        # imageio.imwrite(f"{scene}_pred_diff.png", (np.clip(pred_diff/data_range, 0, 1)*255).astype(np.uint8))
        # imageio.imwrite(f"{scene}_gt_diff.png", (np.clip(gt_diff/data_range, 0, 1)*255).astype(np.uint8))
        # breakpoint()

    keys = perf_scores[0].keys()
    avg_scores = {key: np.mean([score[key] for score in perf_scores]) for key in keys}
    return scene, avg_scores

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Prediction VC')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing prediction')
    parser.add_argument('--modality', type=str, required=True, choices=['normal', 'albedo', 'illum'], help='Modality to evaluate')
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
    with open(osp.join(output_dir, f'{modality}_perf.txt'), 'w') as f:
        mean_val = np.mean([v['mean'] for v in vc_scenes.values()])
        median_val = np.mean([v['median'] for v in vc_scenes.values()])
        rmse_val = np.mean([v['rmse'] for v in vc_scenes.values()])
        a1_val = np.mean([v['a1'] for v in vc_scenes.values()])
        a2_val = np.mean([v['a2'] for v in vc_scenes.values()])
        a3_val = np.mean([v['a3'] for v in vc_scenes.values()])
        a4_val = np.mean([v['a4'] for v in vc_scenes.values()])
        a5_val = np.mean([v['a5'] for v in vc_scenes.values()])

        f.write(f"mean,median,rmse,a1,a2,a3,a4,a5\n")
        f.write(f"{mean_val:.2f},{median_val:.2f},{rmse_val:.2f},{a1_val:.2f},{a2_val:.2f},{a3_val:.2f},{a4_val:.2f},{a5_val:.2f}\n")
        f.write("===================\n")
        f.write("scene_id,metric\n")
        for scene in scenes:
            f.write(f"{scene}: {vc_scenes[scene]}\n")
        print(f"mean,median,rmse,a1,a2,a3,a4,a5\n")
        print(f"{mean_val:.2f},{median_val:.2f},{rmse_val:.2f},{a1_val:.2f},{a2_val:.2f},{a3_val:.2f},{a4_val:.2f},{a5_val:.2f}\n")

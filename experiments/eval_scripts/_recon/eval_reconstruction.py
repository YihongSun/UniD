#!/usr/bin/env python3
"""
Evaluation script for image reconstruction tasks.
Computes PSNR, SSIM, RMSE, and LPIPS metrics.
Parallelized across 8 GPUs.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
import multiprocessing as mp

# Image quality metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# LPIPS
import torch
import lpips


def load_image(image_path):
    """Load image and convert to numpy array in [0, 1] range."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    return img_array


def compute_rmse(pred, gt):
    """Compute Root Mean Squared Error."""
    mse = np.mean((pred - gt) ** 2)
    return np.sqrt(mse)


def compute_psnr(pred, gt):
    """Compute Peak Signal-to-Noise Ratio."""
    return psnr(gt, pred, data_range=1.0)


def compute_ssim(pred, gt):
    """Compute Structural Similarity Index."""
    return ssim(gt, pred, data_range=1.0, channel_axis=2)


def compute_lpips(pred, gt, lpips_model):
    """Compute Learned Perceptual Image Patch Similarity."""
    # Convert from [H, W, 3] in [0, 1] to [1, 3, H, W] in [-1, 1]
    pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float()
    gt_tensor = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float()
    
    # Scale to [-1, 1]
    pred_tensor = pred_tensor * 2 - 1
    gt_tensor = gt_tensor * 2 - 1
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pred_tensor = pred_tensor.cuda()
        gt_tensor = gt_tensor.cuda()
    
    # Compute LPIPS
    with torch.no_grad():
        lpips_value = lpips_model(pred_tensor, gt_tensor)
    
    return lpips_value.item()


def get_file_mapping(split_file, pred_dir, gt_dir, task):
    """
    Parse split file and create mapping between predictions and ground truth.
    
    Args:
        split_file: Path to split file with lines like "ai_003_010/rgb_cam_00_fr0000.png ..."
        pred_dir: Directory containing predictions (pred_rgb_cam_00_fr0000.png)
        gt_dir: Directory containing ground truth (albedo_cam_00_fr0000.png for albedo task)
        task: Task name (e.g., 'albedo', 'depth', 'normal')
    
    Returns:
        List of tuples (pred_path, gt_path)
    """
    file_pairs = []
    
    with open(split_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split the line - first item is the RGB reference
            parts = line.split()
            rgb_path = parts[0]  # e.g., "ai_003_010/rgb_cam_00_fr0000.png"
            
            # Extract scene and frame info
            scene_id = rgb_path.split('/')[0]  # e.g., "ai_003_010"
            filename = rgb_path.split('/')[1].lstrip('rgb_')  # e.g., "rgb_cam_00_fr0000.png" -> "cam_00_fr0000.png"
            
            pred_path = os.path.join(pred_dir, scene_id, f'pred_rgb_{filename}')
            gt_path = os.path.join(gt_dir, scene_id, f'{task}_{filename}')
            
            # Check if both files exist
            if os.path.exists(pred_path) and os.path.exists(gt_path):
                file_pairs.append((pred_path, gt_path))
            else:
                if not os.path.exists(pred_path):
                    print(f"Warning: Prediction not found: {pred_path}")
                if not os.path.exists(gt_path):
                    print(f"Warning: Ground truth not found: {gt_path}")
    
    return file_pairs


def process_gpu_worker(gpu_id, file_pairs_chunk, return_dict):
    """
    Worker function that processes a chunk of file pairs on a specific GPU.
    
    Args:
        gpu_id: GPU device ID
        file_pairs_chunk: List of (pred_path, gt_path) tuples to process
        return_dict: Shared dictionary to store results
    """
    # Set the GPU device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Initialize LPIPS model on this GPU
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()
    
    # Process each file pair
    worker_results = {
        'psnr': [],
        'ssim': [],
        'rmse': [],
        'lpips': []
    }
    
    per_image_results = []
    
    for pred_path, gt_path in tqdm(file_pairs_chunk, desc=f"GPU {gpu_id} processing", position=gpu_id):
        try:
            # Load images
            pred = load_image(pred_path)
            gt = load_image(gt_path)
            
            # Ensure same size
            if pred.shape != gt.shape:
                print(f"Warning: Size mismatch for {pred_path}")
                print(f"  Pred: {pred.shape}, GT: {gt.shape}")
                continue
            
            # Compute metrics
            psnr_value = compute_psnr(pred, gt)
            ssim_value = compute_ssim(pred, gt)
            rmse_value = compute_rmse(pred, gt)
            lpips_value = compute_lpips(pred, gt, lpips_model)
            
            worker_results['psnr'].append(psnr_value)
            worker_results['ssim'].append(ssim_value)
            worker_results['rmse'].append(rmse_value)
            worker_results['lpips'].append(lpips_value)
            
            per_image_results.append({
                'pred_path': pred_path,
                'gt_path': gt_path,
                'psnr': float(psnr_value),
                'ssim': float(ssim_value),
                'rmse': float(rmse_value),
                'lpips': float(lpips_value)
            })
            
        except Exception as e:
            print(f"Error processing {pred_path}: {str(e)}")
            continue
    
    # Store results in shared dictionary
    return_dict[gpu_id] = {
        'metrics': worker_results,
        'per_image': per_image_results
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction quality')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing predicted images')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth images')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Text file listing image pairs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--task', type=str, default='albedo',
                        help='Task name (used to construct GT filename)')
    parser.add_argument('--num_gpus', type=int, default=8,
                        help='Number of GPUs to use for parallel processing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get file mappings
    print(f"Loading file pairs from {args.split_file}...")
    file_pairs = get_file_mapping(args.split_file, args.pred_dir, args.gt_dir, args.task)
    print(f"Found {len(file_pairs)} valid image pairs")
    
    # Split file pairs across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"Using {num_gpus} GPUs for parallel processing")
    
    chunk_size = len(file_pairs) // num_gpus
    file_pair_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(file_pairs)
        file_pair_chunks.append(file_pairs[start_idx:end_idx])
        print(f"GPU {i}: processing {len(file_pair_chunks[i])} images")
    
    # Create shared dictionary for results
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Launch workers
    print("\nStarting parallel processing...")
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_gpu_worker, 
                      args=(gpu_id, file_pair_chunks[gpu_id], return_dict))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("All workers completed!")
    
    # Aggregate results from all workers
    results = {
        'psnr': [],
        'ssim': [],
        'rmse': [],
        'lpips': []
    }
    per_image_results = []
    
    for gpu_id in range(num_gpus):
        worker_data = return_dict[gpu_id]
        for metric in ['psnr', 'ssim', 'rmse', 'lpips']:
            results[metric].extend(worker_data['metrics'][metric])
        per_image_results.extend(worker_data['per_image'])
    
    # Compute average metrics
    avg_metrics = {
        'psnr': float(np.mean(results['psnr'])),
        'ssim': float(np.mean(results['ssim'])),
        'rmse': float(np.mean(results['rmse'])),
        'lpips': float(np.mean(results['lpips'])),
        'num_images': len(results['psnr'])
    }
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of images: {avg_metrics['num_images']}")
    print(f"Average PSNR:     {avg_metrics['psnr']:.4f} dB")
    print(f"Average SSIM:     {avg_metrics['ssim']:.4f}")
    print(f"Average RMSE:     {avg_metrics['rmse']:.4f}")
    print(f"Average LPIPS:    {avg_metrics['lpips']:.4f}")
    print("="*50)
    
    # Save results
    output_file = os.path.join(args.output_dir, 'metrics.json')
    with open(output_file, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nAverage metrics saved to: {output_file}")
    
    # Save per-image results
    per_image_file = os.path.join(args.output_dir, 'per_image_metrics.json')
    with open(per_image_file, 'w') as f:
        json.dump(per_image_results, f, indent=2)
    print(f"Per-image metrics saved to: {per_image_file}")
    
    # Save summary as markdown table
    checkpoint_name = "Model"
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("|                  |   PSNR (dB) |   LPIPS |   SSIM |   RMSE |\n")
        f.write("|------------------|-------------|---------|--------|--------|\n")
        f.write(f"| {checkpoint_name:<16} | {avg_metrics['psnr']:>11.2f} | {avg_metrics['lpips']:>7.2f} | {avg_metrics['ssim']:>6.2f} | {avg_metrics['rmse']:>6.2f} |\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
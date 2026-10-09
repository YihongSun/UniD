from collections import defaultdict
import os
import os.path as osp
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import pycocotools.mask as MaskUtils
import argparse
import multiprocessing as mp
from functools import partial

def init_metrics(num_classes=15):
    """Initialize metric accumulation variables"""
    return {
        'total_area_inter': np.zeros((num_classes,), dtype=np.float64),
        'total_area_union': np.zeros((num_classes,), dtype=np.float64),
        'total_area_gt': np.zeros((num_classes,), dtype=np.float64),
        'total_correct': 0,
        'total_labeled': 0,
        'num_classes': num_classes
    }

def update_metrics(metric_state, gt_mask, pred_mask):
    """
    Update metrics with a single image pair
    
    Args:
        metric_state: dict containing accumulated metrics
        gt_mask: ground truth mask (H, W) with values 0-14 (15=ignore)
        pred_mask: predicted mask (H, W) with values 0-14
    
    Returns:
        metric_state: updated metric state
    """
    num_classes = metric_state['num_classes']
    
    ignore_ind = [0, 22, 28, 54, 55]
    mask = gt_mask != ignore_ind[0]
    for i in ignore_ind[1:]:
        mask &= gt_mask != i
    
    labeled = np.sum(mask)
    correct = np.sum((gt_mask == pred_mask) & mask)
    metric_state['total_correct'] += correct
    metric_state['total_labeled'] += labeled

    unique_classes = np.unique(gt_mask[mask])
    # Compute per-class IoU
    for i in unique_classes:
        gt_i = (gt_mask == i) & mask
        pd_i = (pred_mask == i) & mask
        
        area_inter = np.sum(gt_i & pd_i)
        area_union = np.sum(gt_i | pd_i)
        area_gt = np.sum(gt_i)
        
        metric_state['total_area_inter'][i] += area_inter
        metric_state['total_area_union'][i] += area_union
        metric_state['total_area_gt'][i] += area_gt
    
    return metric_state

def finalize_metrics(metric_state):
    """
    Compute final metrics from accumulated state
    
    Returns:
        metrics: dict with mIoU, fwIoU, PA, mPA
    """
    present = metric_state['total_area_gt'] > 0
    iu = metric_state['total_area_inter'] / (metric_state['total_area_union'] + 1e-10)
    mean_iu = np.mean(iu[present])
    
    freq = metric_state['total_area_gt'] / (np.sum(metric_state['total_area_gt']) + 1e-10)
    fw_iu = (freq[present] * iu[present]).sum()

    acc = metric_state['total_correct'] / (metric_state['total_labeled'] + 1e-10)
    ac = metric_state['total_area_inter'] / (metric_state['total_area_gt'] + 1e-10)
    mean_acc = np.mean(ac[present])

    metrics = {
        'mIoU': float(mean_iu)*100,
        'fwIoU': float(fw_iu)*100,
        'PA': float(acc)*100,
        'mPA': float(mean_acc)*100,
        'per_class_iou': iu.tolist(),
        'per_class_freq': freq.tolist()
    }
    
    return metrics


def process_single_image(fn, data_dir, pred_dir, split, num_classes=57):
    """
    Process a single image and return its metric contribution
    
    Args:
        fn: filename
        data_dir: base data directory
        pred_dir: prediction directory
        split: dataset split
        num_classes: number of classes
    
    Returns:
        metric_state: metrics for this single image
    """
    instance = osp.splitext(fn)[0]
    
    img_path = osp.join(data_dir, 'images', split, fn)
    gt_path = osp.join(data_dir, 'labels', split, instance + '.png')
    pred_path = osp.join(pred_dir, 'pred_' + instance + '.png')
    
    if not osp.exists(img_path):
        raise FileNotFoundError(f"Image file {img_path} not found")
    if not osp.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file {gt_path} not found")
    if not osp.exists(pred_path):
        raise FileNotFoundError(f"Prediction file {pred_path} not found")

    gt_mask = np.array(Image.open(gt_path))
    pred_mask = np.array(Image.open(pred_path))
    
    # Initialize and update metrics for this single image
    metric_state = init_metrics(num_classes=num_classes)
    metric_state = update_metrics(metric_state, gt_mask, pred_mask)
    
    # import imageio
    # imageio.imwrite('tmp.png', imageio.imread(img_path))
    # imageio.imwrite('tmp2.png', (gt_mask / 57 * 255).astype(np.uint8))
    # imageio.imwrite('tmp3.png', (pred_mask / 57 * 255).astype(np.uint8))
    # print(finalize_metrics(metric_state))
    # breakpoint()

    return metric_state


def merge_metric_states(states, num_classes=57):
    """
    Merge multiple metric states from different workers
    
    Args:
        states: list of metric_state dicts
        num_classes: number of classes
    
    Returns:
        merged_state: combined metric state
    """
    merged = init_metrics(num_classes=num_classes)
    
    for state in states:
        merged['total_area_inter'] += state['total_area_inter']
        merged['total_area_union'] += state['total_area_union']
        merged['total_area_gt'] += state['total_area_gt']
        merged['total_correct'] += state['total_correct']
        merged['total_labeled'] += state['total_labeled']
    
    return merged


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Material Segmentation on DMS (Parallel)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted material segmentation masks')
    parser.add_argument('--base_data_dir', type=str, default='data/DMS_v1', help='Path to DMS data directory')
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'], help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./matseg_results', help='Directory to save material segmentation results')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers (default: number of CPU cores)')
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
    data_dir = args.base_data_dir
    split = args.split
    output_dir = args.output_dir
    
    # Determine number of workers
    num_workers = args.num_workers if args.num_workers is not None else mp.cpu_count()
    print(f"Using {num_workers} parallel workers")

    os.makedirs(output_dir, exist_ok=True)

    img_fnames = sorted(os.listdir(osp.join(data_dir, 'images', split)))
    img_fnames = [fn for fn in img_fnames if fn.lower().endswith('.jpg') or fn.lower().endswith('.png') or fn.lower().endswith('.jpeg')]
    print(f"Found {len(img_fnames)} images to evaluate")

    # Create partial function with fixed arguments
    process_func = partial(
        process_single_image,
        data_dir=data_dir,
        pred_dir=pred_dir,
        split=split,
        num_classes=57
    )

    # Process images in parallel
    print("Processing images in parallel...")
    with mp.Pool(processes=num_workers) as pool:
        # Use imap for progress bar
        metric_states = list(tqdm(
            pool.imap(process_func, img_fnames),
            total=len(img_fnames),
            desc="Evaluating Material Segmentation"
        ))

    # Merge results from all workers
    print("Merging results from all workers...")
    merged_state = merge_metric_states(metric_states, num_classes=57)
    
    # Compute final metrics
    metrics = finalize_metrics(merged_state)
    print("Final evaluation metrics:", metrics)
    
    # Save per-class results
    with open(osp.join(output_dir, 'matseg_metrics_per_class.txt'), 'w') as f:
        f.write("class_id,iou\n")
        for i, iou in enumerate(metrics['per_class_iou']):
            f.write(f"{i} ({metrics['per_class_freq'][i]*100:.2f}%): {iou}\n")

    # Save summary
    checkpoint_name = osp.basename(pred_dir)
    summary_file = osp.join(output_dir, 'matseg_metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("|                  |     mIoU    |  fwIoU  |   mPA  |   PA   |\n")
        f.write("|------------------|-------------|---------|--------|--------|\n")
        f.write(f"| {checkpoint_name:<16} | {metrics['mIoU']:>11.2f} | {metrics['fwIoU']:>7.2f} | {metrics['mPA']:>6.2f} | {metrics['PA']:>6.2f} |\n")
    print(f"Summary saved to: {summary_file}")
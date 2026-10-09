import os
import os.path as osp
import json
from tqdm import tqdm
from panopticapi.utils import rgb2id
from PIL import Image
import numpy as np
import cv2
import argparse
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from multiprocessing import Pool, Manager
from functools import partial

def panoptic2boundary(panoptic_mask, panoptic_info, categories):
    ''' Convert panoptic segmentation to boundary map.
    '''
    boundary_contours = []
    instance_seg = np.zeros_like(panoptic_mask, dtype=np.uint8)
    for ann in panoptic_info:
        if categories[ann['category_id']]['isthing']:
            mask = panoptic_mask == ann['id']
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_contours += [contour.flatten().tolist() for contour in contours]
    return boundary_contours

def get_boundary_mask(anno_mask, segments_info, categories):
    ''' Generate boundary mask from panoptic annotation.
    '''
    contours = panoptic2boundary(anno_mask, segments_info, categories)
    b_mask = np.zeros_like(anno_mask, dtype=np.uint8)
    for contour in contours:
        pts = np.array(contour).reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(b_mask, [pts], -1, color=1, thickness=1)
    for ann in segments_info:
        if ann['iscrowd']:
            mask = anno_mask == ann['id']
            b_mask[mask] = 255
    return b_mask

def compute_boundary_metrics(pred, gt, ignore_mask=None, tolerance_factor=0.0075, ignore_border_pixels=0.0):
    pred = np.array(pred, dtype=bool)
    gt = np.array(gt, dtype=bool)
    
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    
    H, W = pred.shape
    diagonal = np.sqrt(H**2 + W**2)
    tolerance = tolerance_factor * diagonal
    border_ignore = int(ignore_border_pixels * diagonal)
    
    # Handle ignore mask and borders
    if ignore_mask is not None:
        ignore_mask = np.array(ignore_mask, dtype=bool)
        pred = pred & ~ignore_mask
        gt = gt & ~ignore_mask
    
    if border_ignore > 0:
        pred[:border_ignore, :] = False
        pred[-border_ignore:, :] = False
        pred[:, :border_ignore] = False
        pred[:, -border_ignore:] = False
        
        gt[:border_ignore, :] = False
        gt[-border_ignore:, :] = False
        gt[:, :border_ignore] = False
        gt[:, -border_ignore:] = False
    
    # Get boundary pixel coordinates
    pred_coords = np.argwhere(pred)  # (N_pred, 2)
    gt_coords = np.argwhere(gt)      # (N_gt, 2)
    
    # Handle edge cases
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return {'tp_pre': 0, 'fp': 0, 'tp_rec': 0, 'fn': 0}
    if len(pred_coords) == 0:
        return {'tp_pre': 0, 'fp': 0, 'tp_rec': 0, 'fn': len(gt_coords)}
    if len(gt_coords) == 0:
        return {'tp_pre': 0, 'fp': len(pred_coords), 'tp_rec': 0, 'fn': 0}
    
    # Use KD-Tree for fast nearest neighbor search (much faster than full distance transform)
    gt_tree = cKDTree(gt_coords)
    pred_tree = cKDTree(pred_coords)
    
    # Query distances for precision (pred -> gt)
    dist_pred_to_gt, _ = gt_tree.query(pred_coords, k=1)
    pred_positives = dist_pred_to_gt <= tolerance
    tp_pre = np.sum(pred_positives)
    fp = len(pred_coords) - tp_pre
    
    # Query distances for recall (gt -> pred)
    dist_gt_to_pred, _ = pred_tree.query(gt_coords, k=1)
    gt_positives = dist_gt_to_pred <= tolerance
    tp_rec = np.sum(gt_positives)
    fn = len(gt_coords) - tp_rec

    return {
        "tp_pre": int(tp_pre),
        "fp": int(fp),
        "tp_rec": int(tp_rec),
        "fn": int(fn)
    }

def process_single_annotation(anno, image_id2anno, categories, data_dir, split, pred_dir, threshold_step):
    """
    Process a single annotation and compute boundary metrics.
    This function is designed to be called in parallel.
    """
    try:
        image_id = anno['image_id']
        file_name = image_id2anno[image_id]['file_name']
        img_path = osp.join(data_dir, split, file_name)
        
        if not osp.exists(img_path):
            return None, f"Image file {img_path} not found"

        anno_path = osp.join(data_dir, 'annotations', f'panoptic_{split}', anno['file_name'])
        if not osp.exists(anno_path):
            return None, f"Annotation file {anno_path} not found"

        anno_mask = rgb2id(np.array(Image.open(anno_path)))
        boundary_mask = get_boundary_mask(anno_mask, anno['segments_info'], categories)
        ignore_mask = (boundary_mask == 255)
        boundary_mask = (boundary_mask == 1)

        pred_path = osp.join(pred_dir, split, "pred_"+osp.splitext(file_name)[0] + '.npy')
        if not osp.exists(pred_path):
            return None, f"Prediction file {pred_path} not found"
        
        pred_boundary = np.load(pred_path)

        metrics = {thrd: compute_boundary_metrics((pred_boundary > thrd), boundary_mask, ignore_mask=ignore_mask) for thrd in np.arange(0, 1, threshold_step)}
        
        return {
            'file_name': file_name,
            'metrics': metrics
        }, None
        
    except Exception as e:
        return None, f"Error processing annotation: {str(e)}"

def summarize_metrics(args, results):
    tally = {thrd: {'tp_pre': [], 'fp': [], 'tp_rec': [], 'fn': []} for thrd in np.arange(0, 1, args.threshold_step)}
    file_names = []
    errors = []
    for result, error in results:
        if error is not None:
            errors.append(error)
        else:
            file_names.append(result['file_name'])
            for thrd in tally.keys():
                for k in tally[thrd].keys():
                    tally[thrd][k].append(result['metrics'][thrd][k])
    
    if errors:
        print(f"\nWarning: {len(errors)} errors occurred during processing:")
        for error in errors[:10]:  # Print first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    final_metrics = {
        thrd: {
            'precision': (np.sum(metrics['tp_pre']) / (np.sum(metrics['tp_pre']) + np.sum(metrics['fp']) + 1e-8)) * 100,
            'recall': (np.sum(metrics['tp_rec']) / (np.sum(metrics['tp_rec']) + np.sum(metrics['fn']) + 1e-8)) * 100,
        } for thrd, metrics in tally.items()
    }
    for thrd in final_metrics.keys():
        precision = final_metrics[thrd]['precision']
        recall = final_metrics[thrd]['recall']
        final_metrics[thrd]['f1'] = 2 * precision * recall / (precision + recall + 1e-8)
    return final_metrics

def write_metrics_to_file(args, final_metrics):
    """ Write the computed metrics to output files.
    """

    with open(osp.join(args.output_dir, f'boundary_odsF_per_thrd_t{args.threshold_step}.txt'), 'w') as f:
        f.write("thrd,precision,recall,f1\n")
        for thrd in final_metrics.keys():
            f.write(f"{thrd:.2f},{final_metrics[thrd]['precision']},{final_metrics[thrd]['recall']},{final_metrics[thrd]['f1']}\n")

    # Save summary metrics
    summary_file = osp.join(args.output_dir, f'boundary_odsF_summary_t{args.threshold_step}.txt')
    best_thrd = max(final_metrics.keys(), key=lambda t: final_metrics[t]['f1'])
    with open(summary_file, 'w') as f:
        f.write("|       THRD       |   Precision |  Recall |   F1   |\n")
        f.write("|------------------|-------------|---------|--------|\n")
        f.write(f"| {best_thrd:^16.2f} | {final_metrics[best_thrd]['precision']:>11.2f} | {final_metrics[best_thrd]['recall']:>7.2f} | {final_metrics[best_thrd]['f1']:>6.2f} |\n")
    print(f"Summary saved to: {summary_file}")
    return best_thrd


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Boundary Detection on COCO')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted boundary masks')
    parser.add_argument('--base_data_dir', type=str, default='data/coco', help='Path to COCO data directory')
    parser.add_argument('--split', type=str, default='val2017', help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./boundary_results', help='Directory to save boundary results')
    parser.add_argument('--threshold_step', type=float, default=0.01, help='Threshold step for binarizing predicted boundary masks')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for processing')
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
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)

    anno_path = osp.join(data_dir, 'annotations', f'panoptic_{split}.json')
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        images = anno['images']
        annotations = anno['annotations']
        categories = {0: {'name': 'void', 'supercategory': 'void'}, **{c['id']: {'name': c['name'], 'supercategory': c['supercategory'], 'isthing': c['isthing']} for c in anno['categories']}}

    image_id2anno = {
        anno['id']: anno for anno in images
    }
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_annotation,
        image_id2anno=image_id2anno,
        categories=categories,
        data_dir=data_dir,
        split=split,
        pred_dir=pred_dir,
        threshold_step=args.threshold_step
    )

    # Process annotations in parallel
    print(f"Processing {len(annotations)} annotations with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, annotations),
            total=len(annotations),
            desc="Evaluating boundaries"
        ))
    
    final_metrics = summarize_metrics(args, results)
    write_metrics_to_file(args, final_metrics)
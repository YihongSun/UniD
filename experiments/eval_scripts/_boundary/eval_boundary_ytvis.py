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
from multiprocessing import Pool, Manager
from functools import partial
import time
import pycocotools.mask as MaskUtils
from eval_boundary import compute_boundary_metrics, summarize_metrics, write_metrics_to_file

def get_boundary_mask_temporal(height, width, f_ind, annos):
    ''' Generate boundary mask from panoptic annotation.
    '''
    b_masks = [np.zeros((height, width), dtype=np.uint8) for _ in f_ind]
    for anno in annos:
        for f_idx in f_ind:
            to_encode = anno['segmentations'][f_idx]
            if to_encode is None:
                continue
            rle = MaskUtils.frPyObjects(to_encode, to_encode['size'][0], to_encode['size'][1])
            segm = MaskUtils.decode(rle)
            contours, _ = cv2.findContours(segm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for pts in contours:
                cv2.drawContours(b_masks[f_ind.index(f_idx)], [pts], -1, color=1, thickness=1)
    return b_masks

def find_constant_negative_regions(bounds, tolerance_factor=0.0075):
    """ Find regions where the boundary predictions are consistently positive across frames.
    Args:
        bounds (np.ndarray): Boundary predictions of shape (T, H, W).
        tolerance_factor (float): Factor to determine the distance tolerance based on image diagonal.
    Returns:
        np.ndarray: A binary mask of shape (H, W) indicating constant positive regions.
    """
    diagonal = np.sqrt(bounds.shape[1]**2 + bounds.shape[2]**2)
    tolerance = tolerance_factor * diagonal

    dist_to_bound = np.stack([distance_transform_edt(~b) > tolerance for b in bounds], axis=0)  # (T, H, W)
    consistent_neg = np.all(dist_to_bound, axis=0)
    return consistent_neg

def process_single_annotation_temporal(info, video_id2anno, categories, data_dir, pred_dir, threshold):
    """
    Process a single annotation and compute boundary metrics.
    This function is designed to be called in parallel.
    """
    video_id, f_start = info
    f_ind = [f_start, f_start + 1, f_start + 2, f_start + 3, f_start + 4]

    vid_anno, annos = video_id2anno[video_id]
    vid_name = vid_anno['file_names'][0].split('/')[0]
    frame_paths = [osp.join(data_dir, 'JPEGImages', f) for f in vid_anno['file_names']]
    for f in frame_paths:
        if not osp.exists(f):
            return None, f"Frame file {f} not found"
    
    height, width = vid_anno['height'], vid_anno['width']
    boundary_masks = get_boundary_mask_temporal(height, width, f_ind, annos)

    pred_paths = [
        osp.join(pred_dir, vid_name, 'boundary', f"{osp.splitext(osp.basename(frame_paths[f]))[0]}_pred_colored.png") for f in f_ind
    ]
    pred_boundaries = [np.asarray(Image.open(p)).mean(2) / 255. for p in pred_paths]
    gt_bounds = [m == 1 for m in boundary_masks]
    pd_bounds = [(pb > threshold) for pb in pred_boundaries]

    consist_gt = find_constant_negative_regions(np.stack(gt_bounds, axis=0))
    consist_pd = find_constant_negative_regions(np.stack(pd_bounds, axis=0))
    overlap = np.logical_and(consist_gt, consist_pd)

    overlap_sum = overlap.sum()
    gt_sum = consist_gt.sum()

    # import imageio
    # imageio.imwrite('tmp.png', imageio.imread(frame_paths[f_start]))
    # imageio.imwrite('tmp1.png', np.hstack([(boundary_masks[ii]*255).astype(np.uint8), (pred_boundaries[ii]*255).astype(np.uint8)]))
    # imageio.imwrite('tmp_consist.png', np.hstack([(consist_gt*255).astype(np.uint8), (consist_pd*255).astype(np.uint8), (overlap*255).astype(np.uint8)]))
    # breakpoint()

    return {
        'file_name': vid_name,
        'f_start': f_start,
        'overlap': overlap_sum,
        'consist_gt': gt_sum,
        'overlap_frac': overlap_sum / (gt_sum + 1e-6)
    }, None


def get_boundary_mask(height, width, annos, f_idx):
    ''' Generate boundary mask from panoptic annotation.
    '''
    b_mask = np.zeros((height, width), dtype=np.uint8)
    for anno in annos:
        to_encode = anno['segmentations'][f_idx]
        if to_encode is None:
            continue
        rle = MaskUtils.frPyObjects(to_encode, to_encode['size'][0], to_encode['size'][1])
        segm = MaskUtils.decode(rle)
        contours, _ = cv2.findContours(segm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for pts in contours:
            cv2.drawContours(b_mask, [pts], -1, color=1, thickness=1)
    return b_mask

def process_single_annotation(info, video_id2anno, categories, data_dir, pred_dir, threshold_step):
    """
    Process a single annotation and compute boundary metrics.
    This function is designed to be called in parallel.
    """
    video_id, eval_f_idx = info
    vid_anno, annos = video_id2anno[video_id]
    vid_name = vid_anno['file_names'][0].split('/')[0]
    frame_paths = [osp.join(data_dir, 'JPEGImages', f) for f in vid_anno['file_names']]
    for f in frame_paths:
        if not osp.exists(f):
            return None, f"Frame file {f} not found"
    pred_paths = [osp.join(pred_dir, vid_name, 'boundary', f"{osp.splitext(osp.basename(f))[0]}_pred_colored.png") for f in frame_paths]
    
    height, width, length = vid_anno['height'], vid_anno['width'], vid_anno['length']

    boundary_mask = get_boundary_mask(height, width, annos, eval_f_idx)
    
    pd_boundary = np.asarray(Image.open(pred_paths[eval_f_idx])).mean(2) / 255.
    gt_boundary = boundary_mask == 1
    
    metrics = {
        thrd: compute_boundary_metrics((pd_boundary > thrd), gt_boundary) for thrd in np.arange(0, 1, threshold_step)
    }
    
    # import imageio
    # imageio.imwrite('tmp.png', imageio.imread(frame_paths[eval_f_idx]))
    # imageio.imwrite('tmp1.png', np.hstack([(boundary_mask*255).astype(np.uint8), (pd_boundary*255).astype(np.uint8)]))
    # breakpoint()

    return {
        'file_name': vid_name,
        'metrics': metrics
    }, None

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Boundary Detection on LVIS')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted boundary masks')
    parser.add_argument('--base_data_dir', type=str, default='data/YTVIS_val', help='Path to YTvis data directory')
    parser.add_argument('--output_dir', type=str, default='./boundary_results', help='Directory to save boundary results')
    parser.add_argument('--threshold_step', type=float, default=0.01, help='threshold_step for binarizing predicted boundary masks')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for processing')
    parser.add_argument('--num_frames', type=int, default=4, help='Number of frames to sample per video for evaluation')
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
    output_dir = args.output_dir
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)

    anno_path = osp.join(data_dir, 'Annotations', 'gt.json')
    with open(anno_path, 'r') as f:
        data = json.load(f)
        videos = data['videos']
        categories = data['categories']
        annotations = data['annotations']

    video_id2anno = {
        anno['id']: (anno, []) for anno in videos
    }
    for anno in annotations:
        if anno['video_id'] in video_id2anno:
            video_id2anno[anno['video_id']][1].append(anno)
    video_ids = list(video_id2anno.keys())



    ### === Standard Evaluation === ###

    info_list = []
    for vid in video_ids:
        vid_anno, _ = video_id2anno[vid]
        indices = np.linspace(0, vid_anno['length'] - 1, args.num_frames, dtype=int)
        info_list += [(vid, idx) for idx in indices]
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_annotation,
        video_id2anno=video_id2anno,
        categories=categories,
        data_dir=data_dir,
        pred_dir=pred_dir,
        threshold_step=args.threshold_step
    )

    print(f"Processing {len(info_list)} info_list with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, info_list),
            total=len(info_list),
            desc="Evaluating boundaries"
        ))
    
    final_metrics = summarize_metrics(args, results)
    best_thrd = write_metrics_to_file(args, final_metrics)


    ### === Temporal Evaluation === ###
    info_list = [(x, max(0, y-4)) for x, y in info_list]
    process_func2 = partial(
        process_single_annotation_temporal,
        video_id2anno=video_id2anno,
        categories=categories,
        data_dir=data_dir,
        pred_dir=pred_dir,
        threshold=best_thrd
    )

    print(f"Processing {len(info_list)} info_list with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func2, info_list),
            total=len(info_list),
            desc="Evaluating boundaries temporal consistency"
        ))
    
    with open(osp.join(output_dir, f'boundary_temp_const_per_video_t{args.threshold_step}_l{args.num_frames}.txt'), 'w') as f:
        f.write("video id,start frame, temporal coverage\n")
        for r in results:
            r = r[0]
            f.write(f"{r['file_name']},{r['f_start']},{r['overlap_frac']}\n")
    
    avg_coverage = np.sum([r[0]['overlap'] for r in results]) / np.sum([r[0]['consist_gt'] for r in results])
    summary_file = osp.join(args.output_dir, f'boundary_temp_const_t{args.threshold_step}.txt')
    with open(summary_file, 'w') as f:
        f.write("threshold, temporal coverage\n")
        f.write(f"{best_thrd:.2f},{avg_coverage}\n")
    print(f"Temporal consistency summary saved to: {summary_file}")
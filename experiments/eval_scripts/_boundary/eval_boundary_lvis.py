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
from eval_boundary import compute_boundary_metrics, summarize_metrics, write_metrics_to_file

def get_boundary_mask(height, width, annos, categories):
    ''' Generate boundary mask from panoptic annotation.
    '''
    b_mask = np.zeros((height, width), dtype=np.uint8)
    for anno in annos:
        for contour in anno['segmentation']:
            pts = np.array(np.array(contour)).reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(b_mask, [pts], -1, color=1, thickness=1)
    return b_mask

def process_single_annotation(image_id, image_id2anno, categories, data_dir, split, pred_dir, threshold_step):
    """
    Process a single annotation and compute boundary metrics.
    This function is designed to be called in parallel.
    """
    try:
        coco_split = {'train': 'train2017', 'val': 'val2017'}[split]
        img_anno, annos = image_id2anno[image_id]
        file_name = osp.basename(img_anno['coco_url'])
        img_path = osp.join(data_dir, coco_split, file_name)
        
        if not osp.exists(img_path):
            return None, f"Image file {img_path} not found"
        
        height, width = img_anno['height'], img_anno['width']

        boundary_mask = get_boundary_mask(height, width, annos, categories)
        ignore_mask = (boundary_mask == 255)
        boundary_mask = (boundary_mask == 1)

        pred_path = osp.join(pred_dir, coco_split, "pred_"+osp.splitext(file_name)[0] + '.npy')
        if not osp.exists(pred_path):
            return None, f"Prediction file {pred_path} not found"
        
        pred_boundary = np.load(pred_path)

        metrics = {thrd: compute_boundary_metrics((pred_boundary > thrd), boundary_mask, ignore_mask=ignore_mask) for thrd in np.arange(0, 1, threshold_step)}

        # import imageio
        # imageio.imwrite('tmp.png', imageio.imread(img_path))
        # imageio.imwrite('tmp1.png', np.hstack([(boundary_mask*255).astype(np.uint8), (np.load(pred_path)*255).astype(np.uint8)]))
        # breakpoint()
        
        return {
            'file_name': file_name,
            'metrics': metrics
        }, None
        
    except Exception as e:
        return None, f"Error processing annotation: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Boundary Detection on LVIS')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted boundary masks')
    parser.add_argument('--base_data_dir', type=str, default='data/coco', help='Path to COCO data directory')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./boundary_results', help='Directory to save boundary results')
    parser.add_argument('--threshold_step', type=float, default=0.01, help='threshold_step for binarizing predicted boundary masks')
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

    anno_path = osp.join(data_dir, 'annotations', f'lvis_v1_{split}.json')
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        images = [img for img in anno['images'] if 'val2017' in img['coco_url']]
        annotations = anno['annotations']
        categories = anno['categories']

    image_id2anno = {
        anno['id']: (anno, []) for anno in images
    }
    for anno in annotations:
        if anno['image_id'] in image_id2anno:
            image_id2anno[anno['image_id']][1].append(anno)
    image_ids = list(image_id2anno.keys())
    
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

    # Process image_ids in parallel
    print(f"Processing {len(image_ids)} image_ids with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, image_ids),
            total=len(image_ids),
            desc="Evaluating boundaries"
        ))
    
    final_metrics = summarize_metrics(args, results)
    write_metrics_to_file(args, final_metrics)
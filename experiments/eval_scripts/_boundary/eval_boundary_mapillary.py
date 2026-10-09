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

def get_boundary_mask(height, width, annos, categories, scale=1.0):
    ''' Generate boundary mask from panoptic annotation.
    '''
    height = int(height * scale)
    width = int(width * scale)
    b_mask = np.zeros((height, width), dtype=np.uint8)
    for anno in annos:
        cat = categories[anno['label']]
        if cat['instances']:
            pts = (np.array(anno['polygon']) * scale).reshape(-1, 1, 2).astype(np.int32) 
            cv2.drawContours(b_mask, [pts], -1, color=1, thickness=1)
    return b_mask

def process_single_annotation(image_id, anno_id2path, categories, data_dir, split, pred_dir, threshold_step, eval_res):
    """
    Process a single annotation and compute boundary metrics.
    This function is designed to be called in parallel.
    """
    
    anno_path = anno_id2path[image_id]
    with open(anno_path, 'r') as f:
        anno_data = json.load(f)
    height, width = anno_data['height'], anno_data['width']
    annos = anno_data['objects']
    if eval_res > 0:
        scale = min(eval_res / max(height, width), 1.0)
    
    boundary_mask = get_boundary_mask(height, width, annos, categories, scale)
    boundary_mask = (boundary_mask == 1)

    pred_path = osp.join(pred_dir, "pred_"+osp.splitext(osp.basename(anno_path))[0] + '.npy')
    pred_boundary = np.load(pred_path)
    if eval_res > 0 and scale != 1.0:
        pred_boundary = cv2.resize(pred_boundary, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    
    # import imageio
    # img_path = anno_path.replace('v2.0/polygons', 'images').replace('.json', '.jpg')
    # imageio.imwrite('tmp.png', imageio.imread(img_path))
    # imageio.imwrite('tmp1.png', np.hstack([(boundary_mask*255).astype(np.uint8), (pred_boundary*255).astype(np.uint8)]))
    
    metrics = {thrd: compute_boundary_metrics((pred_boundary > thrd), boundary_mask) for thrd in np.arange(0, 1, threshold_step)}

    return {
        'file_name': image_id,
        'metrics': metrics
    }, None

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Boundary Detection on Mapillary')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted boundary masks')
    parser.add_argument('--base_data_dir', type=str, default='data/coco', help='Path to COCO data directory')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on')
    parser.add_argument('--version', type=str, default='v2.0', help='Mapillary version')
    parser.add_argument('--output_dir', type=str, default='./boundary_results', help='Directory to save boundary results')
    parser.add_argument('--threshold_step', type=float, default=0.01, help='threshold_step for binarizing predicted boundary masks')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for processing')
    parser.add_argument('--eval_res', type=int, default=0, help='Resolution for evaluation, 0 means original resolution')
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

    with open(osp.join(data_dir, f"config_{args.version}.json"), 'r') as f:
        config = json.load(f)
        categories = config['labels']
        categories = {cat['name']: cat for cat in categories}
    anno_dir = osp.join(data_dir, args.split, args.version, 'polygons')
    anno_ids = [osp.splitext(f)[0] for f in os.listdir(anno_dir) if f.endswith('.json')]
    anno_id2path = {f: osp.join(anno_dir, f+'.json') for f in anno_ids}
    for a in anno_id2path.values():
        assert osp.exists(a), f"Annotation file {a} not found"
    
    # Create partial function with fixed arguments
    process_func = partial(
        process_single_annotation,
        anno_id2path=anno_id2path,
        categories=categories,
        data_dir=data_dir,
        split=split,
        pred_dir=pred_dir,
        threshold_step=args.threshold_step,
        eval_res=args.eval_res
    )

    # Process anno_ids in parallel
    print(f"Processing {len(anno_ids)} anno_ids with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, anno_ids),
            total=len(anno_ids),
            desc="Evaluating boundaries"
        ))
    
    final_metrics = summarize_metrics(args, results)
    write_metrics_to_file(args, final_metrics)
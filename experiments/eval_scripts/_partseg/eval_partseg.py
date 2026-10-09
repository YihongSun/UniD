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
    
    # Create valid mask (exclude ignore regions marked as 15)
    mask = (gt_mask >= 0) & (gt_mask < num_classes)
    
    labeled = np.sum(mask)
    correct = np.sum((gt_mask == pred_mask) & mask)
    metric_state['total_correct'] += correct
    metric_state['total_labeled'] += labeled
    
    # Compute per-class IoU
    for i in range(num_classes):
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
    fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()
    
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Boundary Detection on COCO')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted boundary masks')
    parser.add_argument('--base_data_dir', type=str, default='data/coco', help='Path to COCO data directory')
    parser.add_argument('--split', type=str, default='val2014', help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./boundary_results', help='Directory to save boundary results')
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

    os.makedirs(output_dir, exist_ok=True)

    anno_path = osp.join(data_dir, 'annotations', f'densepose_coco_2014_valminusminival.json')
    with open(anno_path, 'r') as f:
        anno = json.load(f)
        images = anno['images']
        annotations = anno['annotations']

        imgid2anno = defaultdict(list)
        for anno in annotations:
            imgid2anno[anno['image_id']].append(anno)
        imgid2anno = dict(imgid2anno)

    metric_state = init_metrics(num_classes=15)
    for image_anno in tqdm(images):
        image_id = image_anno['id']
        file_name = image_anno['file_name']
        img_path = osp.join(data_dir, split, file_name)
        assert osp.exists(img_path), f"Image file {img_path} not found"

        with Image.open(img_path) as img:
            width, height = img.size
        partseg = np.zeros((height, width), dtype=np.uint8)

        annos = imgid2anno[image_id]
        annos.sort(key=lambda x: int('dp_masks' in x), reverse=True)    # first load annotations w/o densepose masks
        for ann in annos:
            if ann['category_id'] != 1:   # only person class
                continue
            Polys = ann.get('dp_masks', None)
            if Polys is None:
                ignore_mask = MaskUtils.decode(MaskUtils.frPyObjects(ann['segmentation'], height, width))[...,0]
                partseg[ignore_mask>0] = 15 # mark ignored regions
                continue

            local_partseg = np.zeros([256,256])
            for i in range(1,15):
                if(Polys[i-1]):
                    current_mask = MaskUtils.decode(Polys[i-1])
                    local_partseg[current_mask>0] = i
            
            bbr =  np.array(ann['bbox']).astype(int)
            x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]
            x2 = min( [ x2,partseg.shape[1] ] );  y2 = min( [ y2,partseg.shape[0] ] )

            ann_partseg = cv2.resize(local_partseg, (int(x2-x1),int(y2-y1)) ,interpolation=cv2.INTER_NEAREST)
            partseg[y1:y2, x1:x2][ann_partseg>0] = ann_partseg[ann_partseg>0]
        
        pred_path = osp.join(pred_dir, split, 'pred_' + osp.splitext(file_name)[0] + '.png')
        assert osp.exists(pred_path), f"Prediction file {pred_path} not found"

        pred_mask = np.array(Image.open(pred_path))
        metric_state = update_metrics(metric_state, partseg, pred_mask)

        # import imageio
        # imageio.imwrite('tmp.png', (partseg / 15 * 255).astype(np.uint8))
        # imageio.imwrite('tmp2.png', imageio.imread(img_path))
        # imageio.imwrite('tmp3.png', (pred_mask / 15 * 255).astype(np.uint8))
        # print(finalize_metrics(metric_state))
        # breakpoint()
    
    metrics = finalize_metrics(metric_state)
    print("Final evaluation metrics:", metrics)
    with open(osp.join(output_dir, 'partseg_metrics_per_class.txt'), 'w') as f:
        f.write("class_id,iou\n")
        for i, iou in enumerate(metrics['per_class_iou']):
            f.write(f"{i} ({metrics['per_class_freq'][i]*100:.2f}%): {iou}\n")

    checkpoint_name = osp.basename(pred_dir)
    summary_file = osp.join(output_dir, 'partseg_metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("|                  |     mIoU    |  fwIoU  |   mPA  |   PA   |\n")
        f.write("|------------------|-------------|---------|--------|--------|\n")
        f.write(f"| {checkpoint_name:<16} | {metrics['mIoU']:>11.2f} | {metrics['fwIoU']:>7.2f} | {metrics['mPA']:>6.2f} | {metrics['PA']:>6.2f} |\n")
    print(f"Summary saved to: {summary_file}")

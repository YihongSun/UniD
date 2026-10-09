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
from eval_partseg import init_metrics, update_metrics, finalize_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Part Segmentation on Pascal Person')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory containing predicted masks')
    parser.add_argument('--base_data_dir', type=str, default='data/pascal_person', help='Path to PASCAL person part data directory')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./pascal_partseg_results', help='Directory to save boundary results')
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

    
    with open(osp.join(data_dir, 'pascal_person_part_trainval_list', f'{split}_id.txt'), 'r') as f:
        img_ids = list(line.strip() for line in f)

    metric_state = init_metrics(num_classes=7)
    for img_id in tqdm(img_ids, desc="Evaluating on Pascal Person Part"):
        pred_path = osp.join(pred_dir, 'pred_' + img_id + '.png')
        gt_path = osp.join(data_dir, 'pascal_person_part_gt', img_id + '.png')
        assert osp.exists(pred_path), f"Prediction file {pred_path} not found"
        assert osp.exists(gt_path), f"Ground truth file {gt_path} not found"
        
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        assert pred.shape == gt.shape, f"Shape mismatch for {img_id}: pred {pred.shape}, gt {gt.shape}"

        metric_state = update_metrics(metric_state, gt, pred)

        # import imageio
        # imageio.imwrite('tmp_pred.png', np.vstack((pred, gt)).astype(np.uint8)*30)
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

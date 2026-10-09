import os, time, json, glob, sys
import os.path as osp
import argparse
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from dataloaders.datasets.image_datasets.cityscapes import CITYSCAPES_CATEGORIES
from eval_semseg_cityscapes import summarize_metrics

def compute_metrics(gt, pred, categories):
    """ Compute meanIoU, fwIoU, PA, mPA
    """
    num_classes = len(categories)
    total_area_inter = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred = np.zeros((num_classes,), dtype=np.float64)
    total_area_gt = np.zeros((num_classes,), dtype=np.float64)

    img_names = sorted(list(gt.keys()))
    
    for img_name in tqdm(img_names, desc="Computing metrics"):
        gt_path = gt[img_name]
        pred_path = pred[img_name]

        assert img_name in gt_path and img_name in pred_path, f"{img_name} not found in gt/pred paths"

        # Load images
        gt_sem = torch.tensor(np.array(Image.open(gt_path), dtype=np.uint8))
        pd_sem = torch.tensor(np.array(Image.open(pred_path), dtype=np.uint8))
        assert gt_sem.shape == pd_sem.shape, f"{img_name} gt/pred shape mismatch: {gt_sem.shape} vs {pd_sem.shape}"

        # Compute IoU metrics per class
        for i in torch.unique(gt_sem):
            i = int(i.item())
            gt_i = (gt_sem == i)
            pd_i = (pd_sem == i)
            area_inter = torch.sum(gt_i & pd_i)
            area_union = torch.sum(gt_i | pd_i)
            area_gt = torch.sum(gt_i)
            area_pred = torch.sum(pd_i)

            total_area_inter[i] += area_inter.item()
            total_area_union[i] += area_union.item()
            total_area_gt[i] += area_gt.item()
            total_area_pred[i] += area_pred.item()
    
    # Compute final metrics
    eval_cat_ind = [i for i, cat in categories.items() if cat != "unlabeled" and total_area_gt[i] > 0]
    iu = total_area_inter / (total_area_union + 1e-10)
    mean_iu = np.mean(iu[eval_cat_ind]) 

    freq = total_area_gt[eval_cat_ind] / (np.sum(total_area_gt[eval_cat_ind]) + 1e-10)
    fw_iu = (freq * iu[eval_cat_ind]).sum()

    ac = total_area_inter / (total_area_gt + 1e-10)
    mean_acc = np.mean(ac[eval_cat_ind])
    acc = np.sum(total_area_inter[eval_cat_ind]) / (np.sum(total_area_gt[eval_cat_ind]) + 1e-10)

    metrics = {
        'mIoU': mean_iu,
        'fwIoU': fw_iu,
        'PA': acc,
        'mPA': mean_acc,
    }

    aux_data = {
        'per_class_iou': iu,
        'total_area_inter': total_area_inter,
        'total_area_union': total_area_union,
        'total_area_gt': total_area_gt,
        'total_area_pred': total_area_pred,
    }
    return metrics, aux_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on Mapillary validation set for semantic segmentation.")
    parser.add_argument("--pred_dir", type=str, help="Path to the prediction directory.")
    parser.add_argument("--base_data_dir", type=str, help="Path to the eval dataset (Mapillary root).")
    parser.add_argument("--split", type=str, default="val", help="which split to eval (val/test).")
    parser.add_argument("--version", type=str, default="v2.0", help="Mapillary version.")
    parser.add_argument("--output_dir", type=str, default="./quant/", help="output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(osp.join(args.base_data_dir, "coco_annotations", "mapping_info.txt"), "r") as f:
        mappings = json.load(f)
        categories = {int(k): v for k, v in mappings['uni_id2name'].items()}
    # Collect all paths
    gt_paths, pred_paths = dict(), dict()
    gt_base = osp.join(args.base_data_dir, "coco_annotations", args.split)
    for gt_file in os.listdir(gt_base):
            
        if not gt_file.endswith('.png'):
            continue
        
        # Extract base name
        img_name = gt_file.replace('.png', '')
        
        gt_path = osp.join(gt_base, gt_file)
        pred_path = osp.join(args.pred_dir, f"pred_{img_name}.png")

        if not osp.exists(pred_path):
            print(f"Warning: Prediction not found for {img_name} at {pred_path}")
            continue

        gt_paths[img_name] = gt_path
        pred_paths[img_name] = pred_path

    print(f"Found {len(gt_paths)} images to evaluate")
    if len(gt_paths) == 0:
        print("Error: No valid image pairs found!")
        sys.exit(1)

    metrics, aux_data = compute_metrics(gt_paths, pred_paths, categories)
    summarize_metrics(metrics, aux_data, categories, args)
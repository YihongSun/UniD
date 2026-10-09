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
        gt_sem[gt_sem==255] = 0 # Map ignore index to class 0
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
    eval_cat_ind = [i for i, cat in categories.items() if not cat[5]]
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

def summarize_metrics(metrics, aux_data, categories, args):
    # Output keys
    overview_keys = ['mIoU', 'fwIoU', 'mPA', 'PA']

    # Save results
    dataset_name = f"performance_{args.split}"
    out_path = osp.join(args.output_dir, dataset_name + '.csv')
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(','.join(overview_keys) + '\n')
        f.write(','.join(['{:.2f}'.format(metrics[k]*100) for k in overview_keys]) + '\n')
    
    # Save per-class results
    with open(out_path.replace('.csv', '_per_class.csv'), 'w') as f:
        for ii, cat in categories.items():
            gt_perc = aux_data['total_area_gt'][ii] / (np.sum(aux_data['total_area_gt']) + 1e-10)
            iou = aux_data['total_area_inter'][ii] / (aux_data['total_area_union'][ii] + 1e-10)
            pa = aux_data['total_area_inter'][ii] / (aux_data['total_area_gt'][ii] + 1e-10)
            out_str = '{:3d} {:35s}: {:.2f}% - IoU={:.4f} | PA={:.4f}'.format(
                ii, cat[0], gt_perc * 100, iou * 100, pa * 100)
            f.write('{}\n'.format(out_str))
    
    print(f"\nResults Summary:")
    print("=" * 50)
    for k in overview_keys:
        print(f"{k:10s}: {metrics[k]*100:.2f}")
    print("=" * 50)
    print(f"\nSaved results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on Cityscapes validation set for semantic segmentation.")
    parser.add_argument("--pred_dir", type=str, help="Path to the prediction directory.")
    parser.add_argument("--base_data_dir", type=str, help="Path to the eval dataset (Cityscapes root).")
    parser.add_argument("--split", type=str, default="val", help="which split to eval (val/test).")
    parser.add_argument("--output_dir", type=str, default="./quant/", help="output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Collect all paths
    gt_paths, pred_paths = dict(), dict()
    categories = CITYSCAPES_CATEGORIES

    gt_base = osp.join(args.base_data_dir, "gtFine", args.split)
    img_base = osp.join(args.base_data_dir, "leftImg8bit", args.split)
    
    for city in os.listdir(gt_base):
        city_gt_dir = osp.join(gt_base, city)
        if not osp.isdir(city_gt_dir):
            continue
            
        for gt_file in os.listdir(city_gt_dir):
            if not gt_file.endswith('_gtFine_labelIds.png'):
                continue
            
            # Extract base name
            img_name = gt_file.replace('_gtFine_labelIds.png', '')
            
            gt_path = osp.join(city_gt_dir, gt_file)
            pred_path = osp.join(args.pred_dir, city, f"pred_{img_name}_leftImg8bit.png")

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
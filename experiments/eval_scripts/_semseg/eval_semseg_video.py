import os, time, json, glob, sys
import os.path as osp
import argparse
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from dataloaders.datasets.video_datasets.vipseg import VIPSEG_CATEGORIES

def get_color_code(categories):
    color_code = torch.zeros((len(categories),3), dtype=torch.float32)
    for k, v in categories.items():
        color_code[k] = torch.tensor(v['color'], dtype=torch.float32) / 255.0
    return color_code

def compute_vc_metrics(gt, pred, t=4):
    T = gt.shape[0]
    if T < t:
        return None
    vc = []
    for i in range(0, T-t, t):
        gt_common = torch.all(gt[i:i+t] == gt[i:i+1], dim=0)
        pred_common = torch.all(pred[i:i+t] == pred[i:i+1], dim=0)
        num = torch.sum(gt_common & pred_common).item()
        den = torch.sum(gt_common).item()
        vc.append(num / (den + 1e-10))
    return np.mean(vc)


def compute_metrics(gt, pred, categories, early_stop=4):
    """ Compute meanIoU, fwIoU, PA, mPA
    """
    num_classes = len(categories)
    total_area_inter = np.zeros((num_classes,), dtype=np.float64)
    total_area_union = np.zeros((num_classes,), dtype=np.float64)
    total_area_pred = np.zeros((num_classes,), dtype=np.float64)
    total_area_gt = np.zeros((num_classes,), dtype=np.float64)
    total_correct, total_labeled = 0, 0
    total_vcs = {4: [], 8: [], 16: []}

    color_code = get_color_code(categories)
    vid_names = sorted(list(gt.keys()))
    for vid_name in tqdm(vid_names, desc="Computing metrics"):
        gt_paths = gt[vid_name]
        pred_paths = pred[vid_name]

        vcs = {k: {"gt_common": None, "pred_common": None, "gt": None, "pred": None, "values": []} for k in total_vcs.keys()}
        for ii, (gt_p, pd_p) in enumerate(zip(gt_paths, pred_paths)):
            gt_p_base = osp.basename(gt_p)
            pd_p_base = osp.basename(pd_p).replace('_pred_colored.png', '.png')
            assert gt_p_base == pd_p_base, f"{vid_name} gt/pred name mismatch: {gt_p_base} vs {pd_p_base}"
            
            # Load images
            gt_sem = torch.tensor(np.array(Image.open(gt_p), dtype=np.uint8))
            pd_sem_img = torch.tensor(np.array(Image.open(pd_p), dtype=np.float32)) / 255.0
            pd_sem = torch.argmin(torch.norm(pd_sem_img[:,:,None] - color_code[None,None,:,:], p=2, dim=-1), dim=-1)
            pd_sem = pd_sem.to(torch.uint8)
            assert gt_sem.shape == pd_sem.shape, f"{vid_name} gt/pred shape mismatch: {gt_sem.shape} vs {pd_sem.shape}"

            # Compute standard metrics
            mask = (gt_sem > 0) & (gt_sem < num_classes)    # valid mask
            labeled = torch.sum(mask)
            correct = torch.sum((gt_sem == pd_sem) & mask)
            total_correct += correct
            total_labeled += labeled

            # Compute IoU metrics
            for i in torch.unique(gt_sem[mask]):
                gt_i = (gt_sem == i)
                pd_i = (pd_sem == i)
                area_inter = torch.sum(gt_i & pd_i)
                area_union = torch.sum(gt_i | pd_i)
                area_gt = torch.sum(gt_i)
                area_pred = torch.sum(pd_i)

                total_area_inter[i] += area_inter
                total_area_union[i] += area_union
                total_area_gt[i] += area_gt
                total_area_pred[i] += area_pred

            # Compute VC metrics
            for t in vcs.keys():
                if ii == 0:
                    vcs[t]["gt"] = gt_sem
                    vcs[t]["pred"] = pd_sem
                    vcs[t]["gt_common"] = torch.ones_like(gt_sem, dtype=torch.bool)
                    vcs[t]["pred_common"] = torch.ones_like(pd_sem, dtype=torch.bool)
                elif ii % t != 0:
                    vcs[t]["gt_common"] = (vcs[t]["gt"] == gt_sem) & (vcs[t]["gt_common"])
                    vcs[t]["pred_common"] = (vcs[t]["pred"] == pd_sem) & (vcs[t]["pred_common"])
                elif ii % t == 0:
                    gt_common, pd_common = vcs[t]["gt_common"], vcs[t]["pred_common"]
                    score = torch.sum(gt_common & pd_common).item() / (torch.sum(gt_common).item() + 1e-10)
                    vcs[t]["values"].append(score)
                    vcs[t]["gt"] = gt_sem
                    vcs[t]["pred"] = pd_sem
                    vcs[t]["gt_common"] = torch.ones_like(gt_sem, dtype=torch.bool)
                    vcs[t]["pred_common"] = torch.ones_like(pd_sem, dtype=torch.bool)
                else:
                    raise ValueError("This should not happen.")
            if early_stop is not None and ii == early_stop:
                break   # for quick eval

        for t in vcs.keys():
            if len(vcs[t]["values"]) > 0:
                total_vcs[t].append(np.mean(vcs[t]["values"]))
            else:
                total_vcs[t].append(None)

    iu = total_area_inter / (total_area_union + 1e-10)
    mean_iu = np.mean(iu[total_area_gt > 0])  # Only average over classes that appear in GT
    freq = total_area_gt / (np.sum(total_area_gt) + 1e-10)
    fw_iu = (freq[freq > 0] * iu[freq > 0]).sum()
    acc = (total_correct / (total_labeled + 1e-10)).item()
    ac = total_area_inter / (total_area_gt + 1e-10)
    mean_acc = np.mean(ac[total_area_gt > 0])

    vc_4 = np.mean([v for v in total_vcs[4] if v is not None])
    vc_8 = np.mean([v for v in total_vcs[8] if v is not None])
    vc_16 = np.mean([v for v in total_vcs[16] if v is not None])
    metrics = {
        'mIoU': mean_iu,
        'fwIoU': fw_iu,
        'PA': acc,
        'mPA': mean_acc,
        'VC-4': vc_4,
        'VC-8': vc_8,
        'VC-16': vc_16
    }

    aux_data = {
        'per_class_iou': iu,
        'total_area_inter': total_area_inter,
        'total_area_union': total_area_union,
        'total_area_gt': total_area_gt,
        'total_area_pred': total_area_pred,
        'total_correct': int(total_correct),
        'total_labeled': int(total_labeled),
        'total_vcs': total_vcs,
    }
    return metrics, aux_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on video for tracking task.")
    parser.add_argument("--dataset_path", type=str, help="Path to the eval dataset.")
    parser.add_argument("--model", type=str, help="Name of the model.")
    parser.add_argument("--out_dir", type=str, default="./quant/", help="output path.")
    parser.add_argument("--proj", type=str, default="semseg-vipseg", help="which proj to eval.")
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames to evaluate for quick eval.")
    return parser.parse_args()

# python ./quant/eval_scripts/eval_semseg.py --dataset_path data/VIPSeg_val/ --model 'sd2_uni_S_MemAttnTemp(dec)_T4_80000_v1(len=4,st=4,pop_loc=0)'
if __name__ == "__main__":
    args = parse_args()

    print("\n==================================================")
    print(f"Datasets: {args.dataset_path}")
    print(f"Model: {args.model}")

    # collect all paths

    gt_paths, pred_paths = dict(), dict()
    for vid_name in os.listdir(osp.join(args.dataset_path, "imgs")):
        gt_p = sorted(glob.glob(osp.join(args.dataset_path, "sem_masks", vid_name, "*")))
        pd_p = sorted(glob.glob(osp.join('quant', 'video_semseg-vipseg', args.model, "vipseg_val", f"prediction", vid_name, "semseg-vipseg", "*")))

        if len(gt_p) != len(pd_p):
            print(f"Warning: {vid_name} gt/pd mismatch: {len(gt_p)} vs {len(pd_p)} --- IGNORE ---")
            min_len = min(len(gt_p), len(pd_p))
            gt_p = gt_p[:min_len]
            pd_p = pd_p[:min_len]

        gt_paths[vid_name] = gt_p
        pred_paths[vid_name] = pd_p
    
    if 'VIPSeg' in args.dataset_path:
        categories = VIPSEG_CATEGORIES
    else:
        raise NotImplementedError("Only support VIPSeg now.")
    
    metrics, aux_data = compute_metrics(gt_paths, pred_paths, categories, early_stop=args.num_frames)

    overview_keys = ['mIoU', 'fwIoU', 'mPA', 'PA', 'VC-4', 'VC-8', 'VC-16']

    dataset_name = osp.basename(args.dataset_path.strip('/'))
    out_path = osp.join(args.out_dir, args.proj.split('-')[0], args.model, dataset_name+'.csv')
    os.makedirs(osp.dirname(out_path), exist_ok=True)

    with open(out_path, 'w') as f:
        f.write(','.join(overview_keys) + '\n')
        f.write(','.join(['{:.2f}'.format(metrics[k]*100) for k in overview_keys]) + '\n')
    
    with open(out_path.replace('.csv', '_per_class.csv'), 'w') as f:
        for ii, cat in categories.items():
            gt_perc = aux_data['total_area_gt'][ii] / np.sum(aux_data['total_area_gt'])
            iou = aux_data['total_area_inter'][ii] / (aux_data['total_area_union'][ii] + 1e-10)
            pa = aux_data['total_area_inter'][ii] / (aux_data['total_area_gt'][ii] + 1e-10)
            out_str = '{:3d} {:35s}: {:.2f}% - IoU={:.2f} | PA={:.2f}'.format(ii, cat['name'], gt_perc*100, iou*100, pa*100)
            f.write('{}\n'.format(out_str))
    print(f"Saved results to {out_path}")    
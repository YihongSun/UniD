import os
import os.path as osp
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
import multiprocessing as mp
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import find_boundaries
import cv2
from scipy.ndimage import gaussian_filter
import tarfile, io
import sys; sys.path.append("./"); sys.path.append("./experiments/eval_scripts/_normal")
from DSINE.utils import compute_normal_error, compute_normal_metrics
from src.util.alignment import align_depth_least_square

def depth_to_normal(depth, fx, fy, cx, cy, sigma=3.0):
    h, w = depth.shape
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - cx
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - cy

    depth_smooth = gaussian_filter(depth, sigma=sigma)
    Gv, Gu = np.gradient(depth_smooth)

    nx = Gu * fx
    ny = Gv * fy
    nz = -(depth_smooth + v_map * Gv + u_map * Gu)
    normal = np.stack([nx, ny, nz], axis=-1)

    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / (norm + 1e-8)
    normal = normal * -1

    return normal


def get_file_mapping(split_file, pred_dir1, pred_dir2):
    with open(split_file, "r") as f:
        lines = f.readlines()
    
    file_pairs = []
    for line in lines:
        img_name, img_ext = osp.splitext(line.strip().split(' ')[0])
        img_dirname = osp.dirname(img_name)
        img_basename = osp.basename(img_name)

        pred_path1 = osp.join(pred_dir1, img_dirname, "pred_" + img_basename + ".npy")
        pred_path2 = osp.join(pred_dir2, img_dirname, "pred_" + img_basename + ".npy")
        gt_path = line.strip().split(' ')[1]

        if osp.exists(pred_path1) and osp.exists(pred_path2):
            file_pairs.append((pred_path1, pred_path2, gt_path))
        else:
            print(f"Warning: Missing file pair for {img_name} - pred1: {pred_path1}, pred2: {pred_path2}")
    return file_pairs

def process_worker(process_id, files_chunk, return_dict, args):
    metrics_list, fname_list = [], []
    for depth_path, normal_path, gt_path in tqdm(files_chunk, desc=f"Process {process_id}", position=process_id):
        depth = np.load(depth_path)

        tar_obj = tarfile.open(args.tar_file)
        image = tar_obj.extractfile("./" + gt_path).read()
        gt_depth = np.array(Image.open(io.BytesIO(image))) / args.gt_depth_factor
        valid = gt_depth > 0

        depth, scale, shift = align_depth_least_square(
            gt_arr=gt_depth,
            pred_arr=depth,
            valid_mask_arr=gt_depth > 0,
            return_scale_shift=True,
            max_resolution=None,
        )

        normal = np.load(normal_path).transpose((1, 2, 0))
        est_normal = depth_to_normal(depth, args.fx_d, args.fy_d, args.cx_d, args.cy_d)

        est_normal_t = torch.from_numpy(est_normal).float().permute(2, 0, 1).unsqueeze(0)
        normal_t = torch.from_numpy(normal).float().permute(2, 0, 1).unsqueeze(0)
        normal_error = compute_normal_error(est_normal_t, normal_t)
        metric = compute_normal_metrics(normal_error[torch.tensor(valid)[None,None]])

        # print(metric)
        # Image.fromarray(np.hstack([(depth/depth.max()*255).astype(np.uint8),(gt_depth/gt_depth.max()*255).astype(np.uint8)])).save("tmp_depths.png")
        # Image.fromarray(np.hstack([((est_normal+1)/2*255).astype(np.uint8),((normal+1)/2*255).astype(np.uint8)])).save("tmp_normals.png")
        # breakpoint()

        metrics_list.append(metric)
        fname_list.append(osp.basename(depth_path))

    return_dict[process_id] = (metrics_list, fname_list)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Task Consistency between Depth and Normal Predictions')
    parser.add_argument('--d_pred_dir', type=str, required=True, help='Directory containing depth prediction')
    parser.add_argument('--n_pred_dir', type=str, required=True, help='Directory containing normal prediction')
    parser.add_argument('--tar_file', type=str, default='data/marigold_eval/scannet/scannet_val_sampled_800_1.tar', help='Path to dataset directory')
    parser.add_argument('--gt_depth_factor', type=float, default=1000.0, help='Factor to convert depth predictions to meters')
    parser.add_argument('--split_file', type=str, default='Unid/data_split/coco/val2014_partseg.txt', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='quant/ctc_partseg_semseg', help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for parallel evaluation')
    parser.add_argument('--fx_d', type=float, default=577.871, help='Focal length for depth to normal conversion')
    parser.add_argument('--fy_d', type=float, default=577.871, help='Focal length for depth to normal conversion')
    parser.add_argument('--cx_d', type=float, default=319.5, help='Center x for depth to normal conversion')
    parser.add_argument('--cy_d', type=float, default=239.5, help='Center y for depth to normal conversion')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    d_pred_dir = args.d_pred_dir
    n_pred_dir = args.n_pred_dir
    output_dir = args.output_dir
    num_workers = args.num_workers
    assert tarfile.is_tarfile(args.tar_file)

    os.makedirs(output_dir, exist_ok=True)
    
    # Get file mappings
    print(f"Loading file pairs from {args.split_file}...")
    files = get_file_mapping(args.split_file, d_pred_dir, n_pred_dir)
    print(f"Found {len(files)} valid image pairs")

    chunk_size = len(files) // num_workers
    files_chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_workers - 1 else len(files)
        files_chunks.append(files[start_idx:end_idx])
        print(f"Process {i}: processing {len(files_chunks[i])} images")
    
    # process_worker(0, files_chunks[0], {}, args)  # Test the worker function with the first chunk
    manager = mp.Manager()
    return_dict = manager.dict()
    print("\nStarting parallel processing...")
    processes = []
    for process_id in range(num_workers):
        p = mp.Process(target=process_worker, 
                      args=(process_id, files_chunks[process_id], return_dict, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All workers completed!")

    metric_list, fname_list = [], []
    for process_id in range(num_workers):
        metric, fnames = return_dict[process_id]
        metric_list.extend(metric)
        fname_list.extend(fnames)

    mean_error = np.mean([f['mean'] for f in metric_list])
    acc_11_25 = np.mean([f['a3'] for f in metric_list])
    print(f"Average meanError: {mean_error:.2f}")
    print(f"Average acc11.25: {acc_11_25:.2f}")

    with open(osp.join(output_dir, 'ctc_dn_results.txt'), 'w') as f:
        f.write("meanError, Acc11.25\n")
        f.write(f"{mean_error:.2f}, {acc_11_25:.2f}\n")
        f.write("============================\n")
        f.write("Cross-Task Consistency between Partseg and Semseg Predictions\n")
        f.write(f"Partseg Pred Dir: {d_pred_dir}\n")
        f.write(f"Semseg Pred Dir: {n_pred_dir}\n")
        f.write(f"Number of Image Pairs: {len(files)}\n")
        f.write(f"Number of GPUs Used: {num_workers}\n")
        f.write("============================\n")
        f.write("\n\nDetailed results for each image pair:\n")
        f.write("Filename, Metrics\n")
        for fname, metric in zip(fname_list, metric_list):
            f.write(f"{fname}, {metric}\n")

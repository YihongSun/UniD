import os
import os.path as osp
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import torch
import multiprocessing as mp
from skimage.metrics import structural_similarity as ssim
import sys; sys.path.append("./experiments/eval_scripts/_recon")
from eval_reconstruction import get_file_mapping, compute_psnr, compute_lpips
import lpips

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Task Consistency between Albedo and Shading Predictions')
    parser.add_argument('--a_pred_dir', type=str, required=True, help='Directory containing albedo prediction')
    parser.add_argument('--i_pred_dir', type=str, required=True, help='Directory containing illum prediction')
    parser.add_argument('--data_dir', type=str, default='data/hypersim/processed/val', help='Path to dataset directory')
    parser.add_argument('--split_file', type=str, default='Unid/data_split/hypersim/filename_list_val_filtered.txt', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='quant/ctc_albedo_illum', help='Directory to save results')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of worker processes for parallel evaluation')
    args = parser.parse_args()
    return args

def process_gpu_worker(gpu_id, files_chunk, return_dict):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    lpips_model = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_model = lpips_model.cuda()

    psnr_list, lpips_list, fname_list = [], [], []
    for a_p, a_g, i_p, i_g in tqdm(files_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        a_pd = np.array(Image.open(a_p)).astype(np.float32) / 255.0
        i_pd = np.array(Image.open(i_p)).astype(np.float32) / 255.0
        a_gt = np.array(Image.open(a_g)).astype(np.float32) / 255.0
        i_gt = np.array(Image.open(i_g)).astype(np.float32) / 255.0

        comb_gt = np.clip(a_gt * i_gt, 0, 1)
        comb_pred = np.clip(a_pd * i_pd, 0, 1)

        psnr = compute_psnr(comb_pred, comb_gt)
        lpips_score = compute_lpips(comb_pred, comb_gt, lpips_model)
        psnr_list.append(psnr)
        lpips_list.append(lpips_score)
        fname_list.append(osp.basename(a_g))

        # Image.fromarray((comb_gt * 255).astype(np.uint8)).save("tmp_comb_gt.png")
        # Image.fromarray((comb_pred * 255).astype(np.uint8)).save("tmp_comb_pred.png")
        # Image.fromarray((a_pd * 255).astype(np.uint8)).save("tmp_a_pred.png")
        # Image.fromarray((i_pd * 255).astype(np.uint8)).save("tmp_i_pred.png")
        # breakpoint()

    return_dict[gpu_id] = (psnr_list, lpips_list, fname_list)

if __name__ == '__main__':
    args = parse_args()
    data_root = getattr(args, "base_data_dir", getattr(args, "data_dir", None))
    if data_root and not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root is missing: {data_root}. Obtain the dataset from its upstream source "
            "and pass its local path with --base_data_dir or --data_dir."
        )
    a_pred_dir = args.a_pred_dir
    i_pred_dir = args.i_pred_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_gpus = args.num_gpus

    os.makedirs(output_dir, exist_ok=True)
    
    # Get file mappings
    print(f"Loading file pairs from {args.split_file}...")
    file_pairs_albedo = get_file_mapping(args.split_file, a_pred_dir, data_dir, 'albedo')
    file_pairs_illum = get_file_mapping(args.split_file, i_pred_dir, data_dir, 'illum')
    assert len(file_pairs_albedo) == len(file_pairs_illum), f"Number of albedo and illum file pairs do not match: {len(file_pairs_albedo)} vs {len(file_pairs_illum)}"
    print(f"Found {len(file_pairs_albedo)} valid image pairs")

    files = []
    for (a_pred_path, gt_a_path), (i_pred_path, gt_i_path) in zip(file_pairs_albedo, file_pairs_illum):
        assert osp.basename(a_pred_path) == osp.basename(i_pred_path), f"Albedo and Illum file names do not match: {a_pred_path} vs {i_pred_path}"
        files.append((a_pred_path, gt_a_path, i_pred_path, gt_i_path))

    chunk_size = len(files) // num_gpus
    files_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(files)
        files_chunks.append(files[start_idx:end_idx])
        print(f"GPU {i}: processing {len(files_chunks[i])} images")
    
    manager = mp.Manager()
    return_dict = manager.dict()
    print("\nStarting parallel processing...")
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=process_gpu_worker, 
                      args=(gpu_id, files_chunks[gpu_id], return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All workers completed!")

    psnr_list, lpips_list, fname_list = [], [], []
    for gpu_id in range(num_gpus):
        gpu_psnr, gpu_lpips, gpu_fnames = return_dict[gpu_id]
        psnr_list.extend(gpu_psnr)
        lpips_list.extend(gpu_lpips)
        fname_list.extend(gpu_fnames)

    avg_psnr = np.mean(psnr_list)
    avg_lpips = np.mean(lpips_list)
    print(f"Average PSNR: {avg_psnr:.4f} dB")
    print(f"Average LPIPS: {avg_lpips:.4f}")

    with open(osp.join(output_dir, 'ctc_ai_results.txt'), 'w') as f:
        f.write("Cross-Task Consistency between Albedo and Illumination Predictions\n")
        f.write(f"Albedo Pred Dir: {a_pred_dir}\n")
        f.write(f"Illum Pred Dir: {i_pred_dir}\n")
        f.write(f"Data Dir: {data_dir}\n")
        f.write(f"Number of Image Pairs: {len(files)}\n")
        f.write(f"Number of GPUs Used: {num_gpus}\n")
        f.write("============================\n")
        f.write("PSNR (dB), LPIPS\n")
        f.write(f"{avg_psnr:.4f}, {avg_lpips:.4f}\n")
        f.write("\n\nDetailed results for each image pair:\n")
        f.write("Filename, PSNR, LPIPS\n")
        for (a_p, a_g, i_p, i_g), psnr, lpips in zip(files, psnr_list, lpips_list):
            f.write(f"{osp.basename(a_g)}, {psnr:.4f}, {lpips:.4f}\n")
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

def get_file_mapping(split_file, pred_dir1, pred_dir2):
    with open(split_file, "r") as f:
        lines = f.readlines()
    
    file_pairs = []
    for line in lines:
        img_name, img_ext = osp.splitext(line.strip().split(' ')[0])
        img_dirname = osp.dirname(img_name)
        img_basename = osp.basename(img_name)

        pred_path1 = osp.join(pred_dir1, img_dirname, "pred_" + img_basename + ".png")
        pred_path2 = osp.join(pred_dir2, img_dirname, "pred_" + img_basename + ".png")

        if osp.exists(pred_path1) and osp.exists(pred_path2):
            file_pairs.append((pred_path1, pred_path2))
        else:
            print(f"Warning: Missing file pair for {img_name} - pred1: {pred_path1}, pred2: {pred_path2}")
    return file_pairs

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cross-Task Consistency between Partseg and Semseg Predictions')
    parser.add_argument('--p_pred_dir', type=str, required=True, help='Directory containing partseg prediction')
    parser.add_argument('--s_pred_dir', type=str, required=True, help='Directory containing semseg prediction')
    parser.add_argument('--s_person_idx', type=int, default=1, help='Class index for person category in semantic segmentation')
    parser.add_argument('--data_dir', type=str, default='data/coco/val2014', help='Path to dataset directory')
    parser.add_argument('--split_file', type=str, default='Unid/data_split/coco/val2014_partseg.txt', help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='quant/ctc_partseg_semseg', help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for parallel evaluation')
    args = parser.parse_args()
    return args

def process_gpu_worker(process_id, files_chunk, return_dict, args):
    iou_list, contain_list, fname_list = [], [], []
    for partseg_path, semseg_path in tqdm(files_chunk, desc=f"Process {process_id}", position=process_id):
        partseg = np.array(Image.open(partseg_path))
        semseg = np.array(Image.open(semseg_path))

        partseg_human = partseg > 0
        semseg_human = semseg == args.s_person_idx

        if partseg_human.sum() == 0 and semseg_human.sum() == 0:
            iou = 1.0
            contain = 1.0
        elif partseg_human.sum() == 0 or semseg_human.sum() == 0:
            iou = 0.0
            contain = 0.0
        else:
            intersection = np.logical_and(partseg_human, semseg_human).sum()
            union = np.logical_or(partseg_human, semseg_human).sum()
            iou = intersection / union if union > 0 else 1.0

            contain = intersection / partseg_human.sum() if partseg_human.sum() > 0 else 1.0

        # Image.fromarray(partseg_human.astype(np.uint8)*255).save("tmp_partseg_human.png")
        # Image.fromarray(semseg_human.astype(np.uint8)*255).save("tmp_semseg_human.png")
        # breakpoint()

        iou_list.append(iou*100)
        contain_list.append(contain*100)
        fname_list.append(osp.basename(partseg_path))

    return_dict[process_id] = (iou_list, contain_list, fname_list)

if __name__ == '__main__':
    args = parse_args()
    data_root = getattr(args, "base_data_dir", getattr(args, "data_dir", None))
    if data_root and not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root is missing: {data_root}. Obtain the dataset from its upstream source "
            "and pass its local path with --base_data_dir or --data_dir."
        )
    p_pred_dir = args.p_pred_dir
    s_pred_dir = args.s_pred_dir
    data_dir = args.data_dir
    output_dir = args.output_dir
    num_workers = args.num_workers

    os.makedirs(output_dir, exist_ok=True)
    
    # Get file mappings
    print(f"Loading file pairs from {args.split_file}...")
    files = get_file_mapping(args.split_file, p_pred_dir, s_pred_dir)
    print(f"Found {len(files)} valid image pairs")

    chunk_size = len(files) // num_workers
    files_chunks = []
    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_workers - 1 else len(files)
        files_chunks.append(files[start_idx:end_idx])
        print(f"GPU {i}: processing {len(files_chunks[i])} images")
    
    manager = mp.Manager()
    return_dict = manager.dict()
    print("\nStarting parallel processing...")
    processes = []
    for process_id in range(num_workers):
        p = mp.Process(target=process_gpu_worker, 
                      args=(process_id, files_chunks[process_id], return_dict, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("All workers completed!")

    iou_list, contain_list, fname_list = [], [], []
    for process_id in range(num_workers):
        iou, contain, fnames = return_dict[process_id]
        iou_list.extend(iou)
        contain_list.extend(contain)
        fname_list.extend(fnames)

    avg_iou = np.mean(iou_list)
    avg_contain = np.mean(contain_list)
    print(f"Average avgIoU: {avg_iou:.2f}")
    print(f"Average avgcontain: {avg_contain:.2f}")

    with open(osp.join(output_dir, 'ctc_ps_results.txt'), 'w') as f:
        f.write("Cross-Task Consistency between Partseg and Semseg Predictions\n")
        f.write(f"Partseg Pred Dir: {p_pred_dir}\n")
        f.write(f"Semseg Pred Dir: {s_pred_dir}\n")
        f.write(f"Data Dir: {data_dir}\n")
        f.write(f"Number of Image Pairs: {len(files)}\n")
        f.write(f"Number of GPUs Used: {num_workers}\n")
        f.write("============================\n")
        f.write("avgIoU, avgcontain\n")
        f.write(f"{avg_iou:.2f}, {avg_contain:.2f}\n")
        f.write("\n\nDetailed results for each image pair:\n")
        f.write("Filename, IoU, contain\n")
        for fname, iou, contain in zip(fname_list, iou_list, contain_list):
            f.write(f"{fname}, {iou:.2f}, {contain:.2f}\n")
#!/usr/bin/env python3
"""
Multi-GPU inference runner for Unid model.

Usage:
    python Unid/run_batch.py \
        --ckpt_path /path/to/checkpoint.pth \
        --video_dir /path/to/video_frames \
        --output_dir /path/to/output \
        [--gpu_ids 0 1 2 3] \
        -- [optional tags: --post_crf]
Example:
python Unid/run_batch.py --ckpt_path checkpoints/unid_b32_T4_80k --video_dir data/supp_vis/imgs --output_dir vis/ --gpu_ids 0 1 2 3 -- --post_crf
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock
 
PROJS = "depth normal albedo illum boundary partseg-coco matseg semseg-coco"
 
print_lock = Lock()
 
 
def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs, flush=True)
 
 
def run_command(cmd: str, label: str) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    safe_print(f"[{label}] Running: {cmd}")
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        safe_print(f"[{label}] ERROR (code {result.returncode}):\n{result.stderr.strip()}")
    else:
        safe_print(f"[{label}] Done.")
    return result.returncode, result.stdout, result.stderr
 
 
def process_video(
    vid_name: str,
    gpu_id: int,
    ckpt_path: Path,
    video_dir: Path,
    output_dir: Path,
    tags: str,
) -> bool:
    """Run inference + visualization for a single video on a given GPU."""
    ckpt_name = ckpt_path.stem
    vid_input = video_dir / vid_name
    vid_output = output_dir / ckpt_name / vid_name
    vid_video_out = output_dir / f"{ckpt_name}_vids" / f"{vid_name}.mp4"
    label = f"GPU{gpu_id}|{vid_name}"
 
    # --- Step 1: Unid inference ---
    infer_cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python Unid/run.py "
        f"--checkpoint={ckpt_path} "
        f"--input_rgb={vid_input} "
        f"--output_dir={vid_output}"
    )
    if tags:
        infer_cmd += f" {tags}"
 
    rc, _, _ = run_command(infer_cmd, label)
    if rc != 0:
        safe_print(f"[{label}] Inference failed, skipping visualization.")
        return False
 
    # --- Step 2: Visualization ---
    vis_cmd = (
        f"python scripts/vis_results.py "
        f"--image_dir {vid_input} "
        f"--result_dir {vid_output} "
        f"--output_path {vid_video_out} "
        f"--projs {PROJS}"
    )
    rc, _, _ = run_command(vis_cmd, label)
    return rc == 0
 
 
def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Unid inference runner.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory of video frame directories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Root output directory.")
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=None,
        help="CUDA device IDs to use (default: all visible GPUs)."
    )
    parser.add_argument(
        "--tags", type=str, default="",
        help='Additional flags passed to Unid/run.py. Use -- to separate: e.g. -- --post_crf --fp16'
    )
    args, remainder = parser.parse_known_args()
    if remainder:
        remainder = [r for r in remainder if r != "--"]
        args.tags = " ".join(remainder)
 
    ckpt_path = Path(args.ckpt_path).resolve()
    video_dir = Path(args.video_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
 
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not video_dir.exists():
        print(f"ERROR: video_dir not found: {video_dir}", file=sys.stderr)
        sys.exit(1)
 
    # Collect video subdirectories
    videos = sorted([d.name for d in video_dir.iterdir() if d.is_dir()])
    if not videos:
        print(f"No subdirectories found in {video_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(videos)} video(s): {videos}")
 
    # Resolve GPU IDs
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    else:
        # Try to detect from CUDA_VISIBLE_DEVICES env var, else default to [0]
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            gpu_ids = [int(x) for x in cvd.split(",") if x.strip().isdigit()]
        else:
            # Try nvidia-smi
            try:
                result = subprocess.run(
                    "nvidia-smi --query-gpu=index --format=csv,noheader",
                    shell=True, capture_output=True, text=True
                )
                gpu_ids = [int(x) for x in result.stdout.strip().splitlines() if x.strip().isdigit()]
            except Exception:
                gpu_ids = []
        if not gpu_ids:
            gpu_ids = [0]
    print(f"Using GPU IDs: {gpu_ids}")
 
    # Create output directories
    ckpt_name = ckpt_path.stem
    (output_dir / ckpt_name).mkdir(parents=True, exist_ok=True)
    (output_dir / f"{ckpt_name}_vids").mkdir(parents=True, exist_ok=True)
 
    # Distribute videos across GPUs using a shared queue
    gpu_queue: Queue = Queue()
    for vid in videos:
        gpu_queue.put(vid)
 
    results = {}
 
    def worker(gpu_id: int):
        while True:
            try:
                vid_name = gpu_queue.get_nowait()
            except Exception:
                break  # Queue empty
            success = process_video(
                vid_name, gpu_id, ckpt_path, video_dir, output_dir, args.tags
            )
            results[vid_name] = success
            gpu_queue.task_done()
 
    # One thread per GPU; each thread pulls from the shared queue
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gid) for gid in gpu_ids]
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                print(f"Worker raised exception: {exc}", file=sys.stderr)
 
    # Summary
    print("\n=== Summary ===")
    failed = [v for v, ok in results.items() if not ok]
    succeeded = [v for v, ok in results.items() if ok]
    print(f"Succeeded ({len(succeeded)}): {succeeded}")
    if failed:
        print(f"Failed    ({len(failed)}): {failed}")
        sys.exit(1)
    else:
        print("All videos processed successfully.")
 
 
if __name__ == "__main__":
    main()
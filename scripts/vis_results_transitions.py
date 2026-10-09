import os
import os.path as osp
import subprocess
import argparse
import logging
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_base_filename(filename):
    return os.path.splitext(filename)[0]


def find_result_path(result_dir, base):
    expected_path1 = osp.join(result_dir, f"{base}_pred_colored.png")
    expected_path2 = osp.join(result_dir, f"{base}.png")
    if os.path.exists(expected_path1):
        return expected_path1
    if os.path.exists(expected_path2):
        return expected_path2
    return None


def build_source_frame_lists(image_dir, result_dirs):
    image_files = sorted(f for f in os.listdir(image_dir) if not f.startswith('.'))
    bases = [get_base_filename(f) for f in image_files]

    source_frames = [[osp.join(image_dir, f) for f in image_files]]
    for result_dir in result_dirs:
        paths = []
        for base in bases:
            path = find_result_path(result_dir, base)
            if path is None:
                raise FileNotFoundError(
                    f"Missing prediction for '{base}' in {result_dir} "
                    f"(expected {base}_pred_colored.png or {base}.png)"
                )
            paths.append(path)
        source_frames.append(paths)

    lengths = {image_dir: len(image_files)}
    for result_dir in result_dirs:
        n = len([f for f in os.listdir(result_dir) if not f.startswith('.')])
        lengths[result_dir] = n
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Frame count mismatch across directories: {lengths}")

    return source_frames, len(image_files)


def build_segments(T, num_sources, transition_frames):
    num_transitions = num_sources - 1
    total_transition_frames = num_transitions * transition_frames
    remaining = T - total_transition_frames
    if remaining < num_sources:
        raise ValueError(
            f"Not enough frames (T={T}) to fit {num_transitions} transitions of "
            f"{transition_frames} frames each plus {num_sources} non-empty steady segments"
        )

    base_len, extra = divmod(remaining, num_sources)
    steady_lengths = [base_len + (1 if i < extra else 0) for i in range(num_sources)]

    # segments: list of ('steady', source_idx, length) or ('transition', source_idx, length)
    # transition source_idx i means blending source i -> source i+1
    segments = []
    for i in range(num_sources):
        segments.append(('steady', i, steady_lengths[i]))
        if i < num_transitions:
            segments.append(('transition', i, transition_frames))
    return segments


def compose(args):
    result_dirs = [osp.join(args.result_dir, proj) for proj in args.projs]
    source_frames, T = build_source_frame_lists(args.image_dir, result_dirs)
    num_sources = len(source_frames)

    transition_frames = max(1, int(args.transition_period * T))
    segments = build_segments(T, num_sources, transition_frames)

    canonical_size = Image.open(source_frames[0][0]).convert('RGB').size

    output_dir = osp.join(args.result_dir, "_combined_transitions")
    os.makedirs(output_dir, exist_ok=True)

    def load(source_idx, t):
        im = Image.open(source_frames[source_idx][t]).convert('RGB')
        if im.size != canonical_size:
            im = im.resize(canonical_size, Image.Resampling.LANCZOS)
        return im

    t = 0
    with tqdm(total=T) as pbar:
        for kind, source_idx, length in segments:
            for k in range(length):
                if kind == 'steady':
                    frame = load(source_idx, t)
                else:
                    weight = (k + 1) / (transition_frames + 1)
                    im_a = load(source_idx, t)
                    im_b = load(source_idx + 1, t)
                    frame = Image.blend(im_a, im_b, weight)
                frame.save(osp.join(output_dir, f"{t:06d}.png"))
                t += 1
                pbar.update(1)

    assert t == T
    args.output_dir = output_dir


def write_video(args):
    input_pattern = os.path.join(args.output_dir, "*.png")
    subprocess.run([
        "./ffmpeg",
        "-y",
        "-framerate", f'{args.fps}',
        "-pattern_type", "glob",
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-crf", str(args.quality),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        args.output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-fade between input and prediction results over time as a single video")
    parser.add_argument('--image_dir', required=True, help='Directory with original images')
    parser.add_argument('--result_dir', required=True, help='Directory containing per-task prediction subdirectories')
    parser.add_argument('--output_path', default=None, help='output video path')
    parser.add_argument('--projs', required=True, nargs='+', help='List of task subdirectories under result_dir, in transition order')
    parser.add_argument('--transition_period', type=float, default=0.025, help='Fraction of total frames T used for each transition (floored at 1 frame)')
    parser.add_argument('--fps', type=int, default=10, help='FPS for output video')
    parser.add_argument('--quality', type=int, default=17, help='CRF quality for output video (lower is better, default: 20)')
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = args.result_dir.rstrip('/') + "_transitions.mp4"

    compose(args)
    write_video(args)
    logger.info(f"Saved: {args.output_path}")

import os
import os.path as osp
import subprocess
import argparse
from PIL import Image
import math
from tqdm import tqdm

def get_base_filename(filename):
    return os.path.splitext(filename)[0]

def find_matching_files(image_dir, result_dirs, verbose=False):
    files = []
    for fname in os.listdir(image_dir):
        base, _ = os.path.splitext(fname)
        result_paths = []

        for result_dir in result_dirs:
            expected_name = f"{base}_pred_colored.png"
            expected_path1 = os.path.join(result_dir, expected_name)
            expected_path2 = expected_path1.replace('_pred_colored.png', '.png')
            if os.path.exists(expected_path1):
                result_paths.append(expected_path1)
            elif os.path.exists(expected_path2):
                result_paths.append(expected_path2)
            else:
                if verbose:
                    print(f"Missing: {expected_path1} or {expected_path2}")
                break
        if len(result_paths) == len(result_dirs):
            files.append((os.path.join(image_dir, fname), result_paths))
    return files

def arrange_images(images, layout):
    if layout == 'horizontal':
        total_width = sum(i.width for i in images)
        max_height = max(i.height for i in images)
        new_im = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width
    elif layout == 'vertical':
        max_width = max(i.width for i in images)
        total_height = sum(i.height for i in images)
        new_im = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.height
    else:
        cols, rows = layout
        img_width, img_height = images[0].size
        new_im = Image.new('RGB', (cols * img_width, rows * img_height))
        for idx, im in enumerate(images):
            x = (idx % cols) * img_width
            y = (idx // cols) * img_height
            new_im.paste(im, (x, y))
    return new_im

def get_layout(images, base_orientation):
    n = len(images)
    height, width = images[0].height, images[0].width
    asp_ratio = width / height

    if n == 0:
        return (0, 0)
    elif n == 1:
        return (1, 1)
    elif n == 2 or n == 3:
        return 'horizontal' if height > width else 'vertical'
    else:   # n > 3
        best_row, best_cols, min_diff = None, None, float('inf')
        asp_target = 1.5
        # Try all possible number of columns from 1 to n
        for cols in range(1, n + 1):
            rows = math.ceil(n / cols)
            asp_diff = abs((cols * width) / (rows * height) - asp_target)
            if asp_diff < min_diff:
                min_diff = asp_diff
                best_rows = rows
                best_cols = cols
        return (best_cols, best_rows)

def scale_down(image, max_size):
    w, h = image.size
    max_dim = max(w, h)
    if max_dim <= max_size:
        return image
    
    scale = max_size / max_dim

    new_size = (int(w * scale), int(h * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def combine(args):
    result_dirs = [osp.join(args.result_dir, proj) for proj in args.projs]
    matches = find_matching_files(args.image_dir, result_dirs, verbose=args.verbose)
    for image_path, result_paths in tqdm(matches):
        base = get_base_filename(os.path.basename(image_path))

        all_images = [Image.open(image_path).convert('RGB')]
        all_images += [Image.open(p).convert('RGB') for p in result_paths]

        base_orientation = 'portrait' if all_images[0].height > all_images[0].width else 'landscape'
        layout = get_layout(all_images, base_orientation)

        if isinstance(layout, tuple):
            target_size = all_images[0].size
            all_images = [im.resize(target_size, Image.Resampling.LANCZOS) for im in all_images]
            while len(all_images) < layout[0] * layout[1]:
                all_images.append(Image.new('RGB', target_size, (0, 0, 0)))

        combined = arrange_images(all_images, layout)
        combined = scale_down(combined, args.max_size)

        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{base}.png")
        combined.save(output_path)
        if args.verbose:
            print(f"Saved: {output_path}")

        if args.delete_raw:
            for p in result_paths:
                os.remove(p)
    if args.delete_raw:
        for d in result_dirs:
            os.remove(d)

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
    parser = argparse.ArgumentParser(description="Concatenate image and multiple prediction results")
    parser.add_argument('--image_dir', required=True, help='Directory with original images')
    parser.add_argument('--result_dir', required=True, help='List of directories with results like normals, depth, etc.')
    parser.add_argument('--output_path', required=True, help='output video path')
    parser.add_argument('--projs', required=True, nargs='+', help='List of directories with results like normals, depth, etc.')
    parser.add_argument('--verbose', action='store_true', help='Print detailed processing logs')
    parser.add_argument('--delete_raw', action='store_true', help='Delete raw result images after combining')
    parser.add_argument('--fps', type=int, default=24, help='FPS for output video')
    parser.add_argument('--quality', type=int, default=20, help='CRF quality for output video (lower is better, default: 20)')
    parser.add_argument('--max_size', type=int, default=2560, help='Maximum dimension (width or height) for combined images (default: 2000)')
    args = parser.parse_args()
    args.output_dir = osp.join(args.result_dir, "_combined")

    combine(args)
    write_video(args)

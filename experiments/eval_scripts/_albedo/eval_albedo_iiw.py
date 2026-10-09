import os
import os.path as osp
import sys
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

from multiprocessing import Pool, Manager
from functools import partial


def compute_whdr(reflectance, judgements, delta=0.10):
    """ Return the WHDR score for a reflectance image, evaluated against human
    judgements.  The return value is in the range 0.0 to 1.0, or None if there
    are no judgements for the image.  See section 3.5 of our paper for more
    details.

    :param reflectance: a numpy array containing the linear RGB
    reflectance image.

    :param judgements: a JSON object loaded from the Intrinsic Images in
    the Wild dataset.

    :param delta: the threshold where humans switch from saying "about the
    same" to "one point is darker."
    """

    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    weight_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
            int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
            int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker != alg_darker:
            error_sum += weight
        weight_sum += weight

    if weight_sum:
        return error_sum / weight_sum
    else:
        return None


def process_single_image(id, args, delta=0.10):
    reflect_path = osp.join(args.pred_dir, f'pred_{id}.png')
    gt_path = osp.join(args.gt_dir, "annotations", f'{id}.json')
    reflectance = np.asarray(Image.open(reflect_path)).astype(float) / 255.0
    judgements = json.load(open(gt_path))
    whdr = compute_whdr(reflectance, judgements, delta)
    return whdr

def main():
    parser = argparse.ArgumentParser(description='Evaluate reconstruction quality')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing predicted images')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth images')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Text file listing image pairs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of GPUs to use for parallel processing')
    
    args = parser.parse_args()

    # Read image IDs from split file
    with open(args.split_file, 'r') as f:
        lines = [x.split('.png')[0] for x in f.readlines()]
    
    # Process images
    func5 = partial(process_single_image, args=args, delta=0.05)
    func10 = partial(process_single_image, args=args, delta=0.10)
    func15 = partial(process_single_image, args=args, delta=0.15)
    results = [[func5(x), func10(x), func15(x)] for x in tqdm(lines)]
    results = np.array(results) * 100
    whdr5, whdr10, whdr15 = np.mean(results, axis=0)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save summary metrics
    with open(osp.join(args.output_dir, f'whdr_per_id.txt'), 'w') as f:
        f.write("id,whdr5,whdr10,whdr15\n")
        for id, (w5, w10, w15) in zip(lines, results):
            f.write(f"{id},{w5:.2f},{w10:.2f},{w15:.2f}\n")

    summary_file = osp.join(args.output_dir, f'whdr.txt')
    with open(summary_file, 'w') as f:
        f.write("|       WHDR       |     whdr5   |  func10 |  func15 |\n")
        f.write("|------------------|-------------|---------|---------|\n")
        f.write(f"|                  | {whdr5:>11.2f} | {whdr10:>7.2f} | {whdr15:>6.2f} |\n")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()

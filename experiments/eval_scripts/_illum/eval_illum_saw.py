import argparse
import csv
import os
import numpy as np
from scipy.ndimage import maximum_filter, sobel
from tqdm import tqdm
import itertools
import json
from PIL import Image, ImageDraw

def plot_and_save_2D_arrays(filename, arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    """ Plots multiple arrays in the same plot based on the specifications and
    saves it. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_2D_arrays(arrs, title, xlabel, xinterval, ylabel, yinterval, line_names, simplified)
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

def plot_2D_arrays(arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[], simplified=False):
    """ Plots multiple arrays in the same plot based on the specifications. """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.clf()

    for i, arr in enumerate(arrs):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                'The array should be 2D and the second dimension should be 2!'
                ' Shape: %s' % str(arr.shape)
            )

        # Plot last one with black
        if i == len(arrs) - 1:
            plt.plot(arr[:, 0], arr[:, 1], color='black')
        else:
            plt.plot(arr[:, 0], arr[:, 1])

    # If simplified, we don't show text anywhere
    if not simplified:
        plt.title(title[:30])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if line_names:
            plt.legend(line_names, loc=6, bbox_to_anchor=(1, 0.5))

    if xinterval:
        plt.xlim(xinterval)
    if yinterval:
        plt.ylim(yinterval)

    plt.tight_layout()

def gen_class_weights_str(class_weights):
    assert len(class_weights) == 3 or class_weights is None
    if class_weights is None:
        suffix = 'None'
    else:
        suffix = '-'.join([
            '%.2f' % cw
            for cw in class_weights
        ])

    return 'class_weights-%s' % suffix

def to_perc(x):
    return '%.2f%%' % (x * 100)

def ensuredir(dirpath):
    if os.path.exists(dirpath):
        return

    try:
        os.makedirs(dirpath)
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(dirpath):
            pass
        else:
            raise

def load_shading_image_arr(pred_shading_dir, photo_id):
    """ Loads up the decomposed shading layer for a photo as a numpy array with
    values in [0, 1]. """

    shading_img_path = os.path.join(pred_shading_dir, 'pred_%s.png' % photo_id) # %s-s.png
    if not os.path.exists(shading_img_path):
        raise ValueError('Could not find decomposed shading image at "%s"' % shading_img_path)

    shading_image = Image.open(shading_img_path)
    shading_image_arr = np.asarray(shading_image).astype(float) / 255.0
    return shading_image_arr

def load_pixel_labels(pixel_labels_dir, photo_id):
    """ Loads up the ground truth pixel labels for a photo as a numpy array. """

    pixel_labels_path = os.path.join(pixel_labels_dir, '%s.npy' % photo_id)
    if not os.path.exists(pixel_labels_path):
        raise ValueError('Could not find ground truth labels at "%s"' % pixel_labels_path)

    return np.load(pixel_labels_path)

def load_photo_ids_for_split(splits_dir, dataset_split):
    """ Loads photo ids in a SAW dataset split. """
    split_name = {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    photo_ids_path = os.path.join(splits_dir, '%s_ids.npy' % split_name)
    return np.load(photo_ids_path)

def compute_gradmag(image_arr):
    """ Compute gradient magnitude image of a 2D (grayscale) image. """
    assert image_arr.ndim == 2
    dy = sobel(image_arr, axis=0)
    dx = sobel(image_arr, axis=1)
    return np.hypot(dx, dy)

def srgb_to_rgb(srgb):
    """ Convert an image from sRGB to linear RGB.

    :param srgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret

def grouped_confusion_matrix(y_true, y_pred):
    """
    Create "grouped" (3x2) confusion matrix from ground truth and predicted labels.
    The baselines predict only two types of labels (non-smooth/smooth shading),
    but we have 3 types of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    # Sanity checks
    assert set(np.unique(y_true)).issubset(set([0, 1, 2]))
    assert set(np.unique(y_pred)).issubset(set([0, 1]))
    assert len(y_pred) == len(y_true)
    assert y_true.ndim == 1
    assert y_pred.ndim == 1

    conf_mx = np.zeros((3, 2), dtype=int)
    for gt_label in range(3):
        mask = y_true == gt_label
        for pred_label in range(2):
            conf_mx[gt_label, pred_label] = np.sum(y_pred[mask] == pred_label)

    return conf_mx

def get_pr_from_conf_mx(conf_mx, class_weights):
    """
    Compute precision and recall based on a special 3x2 confusion matrix with
    class reweighting.
    The input is not a proper confusion matrix, because the baselines predict
    only two types of labels (non-smooth/smooth shading), but we have 3 types
    of ground truth labels:
        (0) normal/depth discontinuity non-smooth shading (NS-ND)
        (1) shadow boundary non-smooth shading (NS-SB)
        (2) smooth shading (S)
    Ground truth labels 0, 1 are mapped to predicted label 0 (non-smooth shading).
    Ground truth label 2 is mapped to predicted label 1 (smooth shading).
    """
    assert not np.all(conf_mx == 0)
    assert conf_mx.ndim == 2
    assert conf_mx.shape[0] == 3
    assert conf_mx.shape[1] == 2

    # Rebalance confusion matrix rows
    if class_weights:
        assert len(class_weights) == 3
        label_counts = np.sum(conf_mx, axis=1)
        assert np.all(label_counts > 0)
        conf_mx = conf_mx.astype(float)
        conf_mx *= (np.array(class_weights, dtype=float) / label_counts)[:, np.newaxis]

    smooth_count_true = np.sum(conf_mx[2, :])
    smooth_count_pred = np.sum(conf_mx[:, 1])
    smooth_count_correct = float(conf_mx[2, 1])
    assert smooth_count_true != 0
    smooth_recall = smooth_count_correct / smooth_count_true
    if smooth_count_pred:
        smooth_prec = smooth_count_correct / smooth_count_pred
    else:
        smooth_prec = 1

    return smooth_prec, smooth_recall

def generate_pr(saw_image_dir, pixel_labels_dir, splits_dir, out_dir,
                dataset_split, class_weights, bl_filter_size, algo_configs,
                thres_count=200):
    """
    Generate precision-recall curves for each specified algorithm.

    :param saw_image_dir: Directory which contains the SAW images (input to the
    CNN).

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param splits_dir: Directory which contains the list of photo IDs for each
    dataset split (training, validation, test).

    :param out_dir: Directory where we will save the generated PR curves.

    :param dataset_split: Dataset split we want to evaluate on. Can be "R"
    (training), "V" (validation) or "E" (test).

    :param class_weights: List of weights for the 3 classes (NS-ND, NS-SB, S).
    We used [1, 1, 2] in the paper.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.

    :param algo_configs: List of baselines as (algorithm slug, predicted
    (decomposed) shading directory) pairs or ("saw_pixelnet",
    "path_to_trained_net_dir") for our trained CNN.

    :param thres_count: Number of thresholds we want to evaluate on. Check
    ``gen_pr_thres_list`` to see how we sample thresholds between 0 and 1.
    """
    bl_names_dic = {
        'baseline_reflectance': 'Constant R',
        'zhou2015_reflprior': '[Zhou et al. 2015]',
        'bell2014_densecrf': '[Bell et al. 2014]',
        'grosse2009_color_retinex': 'Color Retinex',
        'grosse2009_grayscale_retinex': 'Grayscale Retinex',
        'zhao2012_nonlocal': '[Zhao et al. 2012]',
        'garces2012_clustering': '[Garces et al. 2012]',
        'shen2011_optimization': '[Shen et al. 2011]',
        'saw_pixelnet': 'SAW Pixelnet',
    }

    rootdir = out_dir
    ensuredir(rootdir)
    thres_list = gen_pr_thres_list(thres_count)
    photo_ids = load_photo_ids_for_split(
        splits_dir=splits_dir, dataset_split=dataset_split
    )

    plot_arrs = []
    line_names = []

    fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
    title = '%s Precision-Recall' % (
        {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
    )

    def snap_plot():
        plot_and_save_2D_arrays(
            filename=os.path.join(rootdir, fn) + '.pdf', arrs=plot_arrs, title=title,
            xlabel='Recall', xinterval=(0, 1), ylabel='Precision',
            yinterval=(0, 1), line_names=line_names,
        )
        save_plot_arr_to_csv(
            file_path=os.path.join(rootdir, fn) + '.csv',
            thres_list=thres_list,
            arrs=plot_arrs,
            line_names=line_names,
        )

    for algo_slug, algo_dir in algo_configs:
        print(algo_slug, algo_dir)
        eval_kwargs = dict(
            pred_shading_dir=algo_dir,
            bl_filter_size=bl_filter_size,
        )
        rdic_list = get_precision_recall_list(
            pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_ids=photo_ids, class_weights=class_weights,
            eval_func=eval_baseline_on_photo, eval_kwargs=eval_kwargs,
        )

        plot_arrs.append(gen_plot_arr(rdic_list))
        if algo_slug in bl_names_dic:
            line_names.append(bl_names_dic[algo_slug])
        else:
            line_names.append('%s, bfs (%s)' % (
                algo_slug, bl_filter_size
            ))
        snap_plot()

def gen_plot_arr(rdic_list):
    """ Generate a list of x, y pairs based on recall and precision for each
    threshold. """
    # x: recall, y: precision
    plot_arr = np.empty((len(rdic_list), 2))
    for i, rdic in enumerate(rdic_list):
        plot_arr[i, 0] = rdic['overall_recall']
        plot_arr[i, 1] = rdic['overall_prec']

    return plot_arr

def save_plot_arr_to_csv(file_path, thres_list, arrs, line_names):
    """ Save plot arrays (each is a list of x,y pairs) as a CSV file. """
    assert len(arrs) == len(line_names)
    assert len(thres_list) == arrs[0].shape[0]

    csv_data = []
    header = ['Threshold', 'Recall', 'Precision']

    for ln, arr in zip(line_names, arrs):
        csv_data.append([])
        csv_data.append([ln])
        csv_data.append(header)

        for thres, (recall, prec) in zip(thres_list, arr):
            csv_data.append([
                '%.3f' % thres,
                to_perc(recall),
                to_perc(prec),
            ])

    with open(file_path, 'w') as fp:
        wr = csv.writer(fp, delimiter=',')
        wr.writerows(csv_data)
    
    from scipy.interpolate import interp1d
    recall = arrs[0][:, 0]
    precision = arrs[0][:, 1]
    sort_idx = np.argsort(recall)
    recall_sorted = recall[sort_idx]
    precision_sorted = precision[sort_idx]
    precision_func = interp1d(recall_sorted, precision_sorted,kind='linear',bounds_error=False,fill_value=(precision_sorted[0], precision_sorted[-1]))

    p_03 = precision_func(0.3) * 100
    p_05 = precision_func(0.5) * 100
    p_07 = precision_func(0.7) * 100
    ap = np.trapz(precision_sorted, recall_sorted) * 100
    with open(file_path.replace('.csv', '_metrics.txt'), 'w') as f:
        f.write("p@0.3,p@0.5,p@0.7,AP\n")
        f.write(f"{p_03:.2f},{p_05:.2f},{p_07:.2f},{ap:.2f}\n")


def gen_pr_thres_list(thres_count):
    """ Generate a list of thresholds between 0 and 1, generating more around 0
    and 1 than in the middle. """
    zero_to_one = np.logspace(-4, 0, num=thres_count//2)
    h0 = zero_to_one / 2
    h1 = 1 - h0
    thres_list = sorted(list(h0) + list(h1))

    return thres_list

def get_precision_recall_list(pixel_labels_dir, thres_list, photo_ids,
                              class_weights, eval_func, eval_kwargs):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` over all photos.

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_ids: IDs of the photos we want to evaluate on.

    :param class_weights: List of weights for the 3 classes (NS-ND, NS-SB, S).
    We used [1, 1, 2] in the paper.

    :param eval_func: Evaluation function which returns a confusion matrix for
    a given photo.

    :param eval_kwargs: Extra parameters for the evaluation fuction.

    """

    output_count = len(thres_list)
    overall_conf_mx_list = [
        np.zeros((3, 2), dtype=int)
        for _ in range(output_count)
    ]

    for photo_id in tqdm(photo_ids):
        conf_mx_list = eval_func(
            pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_id=photo_id, **eval_kwargs
        )

        for i, conf_mx in enumerate(conf_mx_list):
            # If this image didn't have any labels
            if conf_mx is None:
                continue

            overall_conf_mx_list[i] += conf_mx

    ret = []
    for i in range(output_count):
        overall_prec, overall_recall = get_pr_from_conf_mx(
            conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
        )

        ret.append(dict(
            overall_prec=overall_prec,
            overall_recall=overall_recall,
            overall_conf_mx=overall_conf_mx_list[i],
        ))

    return ret

def eval_baseline_on_photo(pixel_labels_dir, thres_list, photo_id,
                           pred_shading_dir, bl_filter_size):
    """
    This method generates a list of precision-recall pairs and confusion
    matrices for each threshold provided in ``thres_list`` for a specific
    photo.

    :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

    :param thres_list: List of shading gradient magnitude thresholds we use to
    generate points on the precision-recall curve.

    :param photo_id: ID of the photo we want to evaluate on.

    :param pred_shading_dir: Directory which contains the intrinsic image
    decompositions for all photos generated by a decomposition algorithm.

    :param bl_filter_size: The size of the maximum filter used on the shading
    gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
    """
    shading_image_arr = load_shading_image_arr(
        pred_shading_dir=pred_shading_dir, photo_id=photo_id
    )
    shading_image_linear = srgb_to_rgb(shading_image_arr)
    shading_image_linear_grayscale = np.mean(shading_image_linear, axis=2)
    shading_gradmag = compute_gradmag(shading_image_linear_grayscale)

    if bl_filter_size:
        shading_gradmag = maximum_filter(shading_gradmag, size=bl_filter_size)

    # We have the following ground truth labels:
    # (0) normal/depth discontinuity non-smooth shading (NS-ND)
    # (1) shadow boundary non-smooth shading (NS-SB)
    # (2) smooth shading (S)
    # (100) no data, ignored
    y_true = load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)
    y_true = np.ravel(y_true)
    ignored_mask = y_true == 100

    # If we don't have labels for this photo (so everything is ignored), return
    # None
    if np.all(ignored_mask):
        return [None] * len(thres_list)

    ret = []
    for thres in thres_list:
        y_pred = (shading_gradmag < thres).astype(int)
        y_pred = np.ravel(y_pred)
        # Note: y_pred should have the same image resolution as y_true
        assert y_pred.shape == y_true.shape
        ret.append(grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask]))

    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel_labels_dir', default='data/saw/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0', help='Path to the directory which contains the ground truth labels.')
    parser.add_argument('--saw_image_dir', default='data/saw/saw_images_512', help='Path to the directory which contains the SAW photos.')
    parser.add_argument('--saw_anno_dir', default='data/saw/saw_annotations_json', help='Path to the directory which contains the SAW annotations.')
    parser.add_argument('--splits_dir', default='data/saw/saw_splits', help='Path to the directory which contains the dataset splits.')
    parser.add_argument('--pred_dir', required=True, help='Local directory containing the reference intrinsic-image decompositions.')
    parser.add_argument('--out_dir', default='out_dir', help='Path to the directory where we will save the generated precision-recall curves.')
    parser.add_argument('--dataset_split', default='E', help='Which dataset split we should use: "R" (training), "V" (validation), "E" (test).')
    args = parser.parse_args()
    for option in ("pixel_labels_dir", "saw_image_dir", "saw_anno_dir", "splits_dir", "pred_dir"):
        value = getattr(args, option)
        if not os.path.isdir(value):
            raise FileNotFoundError(
                f"Required local path for --{option}: {value}. Obtain the dataset or reference predictions "
                "from their upstream sources and pass the local directory."
            )

    generate_pr(
        saw_image_dir=args.saw_image_dir,
        pixel_labels_dir=args.pixel_labels_dir,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        dataset_split=args.dataset_split,
        class_weights=[1, 1, 2],
        bl_filter_size=10,
        algo_configs=[('ours', args.pred_dir)],
    )


if __name__ == '__main__':
    main()

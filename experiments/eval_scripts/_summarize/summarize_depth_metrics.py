import os
import re
import argparse
import pandas as pd
from tabulate import tabulate

# Define name mappings
dataset_shortnames = {
    "nyu_test_full": "NYUv2",
    "kitti_eigen_test_full": "KITTI",
    "eth3d_full": "ETH3D",
    "scannet_val_800_1": "ScanNet",
    "diode_val_all": "DIODE",
}

metric_shortnames = {
    "abs_relative_difference": "AR", # AbsRel
    "delta1_acc": "d1",
    # Add more if needed
}

def format_method_col(method_name, width=25):
    return method_name[:width].ljust(width)

def parse_eval_file(file_path, target_metrics):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    dataset = None
    method = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(file_path))))
    headers, values = [], []

    for i, line in enumerate(lines):
        if 'on dataset:' in line:
            dataset = line.split('on dataset:')[-1].strip()
        if line.startswith('abs_relative_difference'):
            headers = re.split(r'\s{2,}', line.strip())
            values = re.split(r'\s{2,}', lines[i+1].strip())
            break

    if dataset and headers and values:
        metrics = dict(zip(headers, values))
        return method, dataset, {f"{dataset}/{k}": float(metrics[k]) for k in target_metrics if k in metrics}
    return None, None, {}

def main(eval_root, methods, datasets, target_metrics, alignment):
    results = {}

    for method in methods:
        for dataset in datasets:
            file_path = os.path.join(
                eval_root,
                method,
                dataset,
                f"eval_metrics-{alignment}.txt"
            )

            if not os.path.isfile(file_path):
                print(f"Warning: file not found: {file_path}")
                continue

            method_name, dataset_name, metric_vals = parse_eval_file(file_path, target_metrics)
            if method_name and metric_vals:
                if method_name not in results:
                    results[method_name] = {}
                results[method_name].update(metric_vals)

    # Create and display DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Method'
    # df = df.sort_index(axis=1)

    # # To percentage
    df = df * 100
    
    # Rename columns 
    def shorten_col(col):
        if "/" in col:
            dataset, metric = col.split("/", 1)
            dataset_short = dataset_shortnames.get(dataset, dataset)
            metric_short = metric_shortnames.get(metric, metric)
            return f"{dataset_short}/{metric_short}"
        return col

    df.columns = [shorten_col(col) for col in df.columns]
    # print(df)

    # Apply fixed-width formatting to method names
    fixed_width_index = [format_method_col(name, width=50) for name in df.index]
    df.index = fixed_width_index

    # Print aligned table
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".2f"))

    # Optional: save
    # df.to_csv("eval_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize evaluation metrics across methods and datasets.")
    parser.add_argument("--methods", nargs='+', required=True, help="List of method folder names")
    parser.add_argument("--eval_root", default="./quant/depth", help="Root directory containing method folders")
    parser.add_argument("--alignment", default="least_square", help="alignment mode")
    parser.add_argument("--datasets", nargs='+', default=["nyu_test", "kitti_eigen_test", "eth3d", "scannet", "diode"], help="List of datasets to look for within each method folder")
    parser.add_argument("--target_metrics", nargs='+', default=["abs_relative_difference", "delta1_acc"], help="List of metrics to extract")

    args = parser.parse_args()
    main(args.eval_root, args.methods, args.datasets, args.target_metrics, args.alignment)

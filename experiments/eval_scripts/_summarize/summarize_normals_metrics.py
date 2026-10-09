import os
import re
import argparse
import pandas as pd
from tabulate import tabulate

# Define name mappings
dataset_shortnames = {
    "nyuv2": "NYUv2",
    "scannet": "ScanNet",
    "ibims": "iBims-1",
    "sintel": "Sintel",
    "oasis": "OASIS",
}

metric_shortnames = {
    "mean": "Mean", # AbsRel
    "11.25": "11.25",
    # Add more if needed
}

target_metrics_indices = {
    "mean": 0,
    "median": 1,
    "rmse": 2,
    "5": 3,
    "7.5": 4,
    "11.25": 5,
    "22.5": 6,
    "30": 7,
}

def format_method_col(method_name, width=25):
    return method_name[:width].ljust(width)

def parse_normals_file(file_path, target_metrics):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Look for the line that contains the actual values
    for line in lines:
        if line.strip() and line.strip()[0].isdigit():
            values = list(map(float, line.strip().split()))
            return {metric: values[target_metrics_indices[metric]] for metric in target_metrics}
    return {}

def main(eval_root, methods, datasets, target_metrics):
    results = {}

    for method in methods:
        method_results = {}
        for dataset in datasets:
            file_path = os.path.join(
                eval_root,
                method,
                "test",
                dataset,
                "metrics.txt"
            )

            if not os.path.isfile(file_path):
                print(f"Warning: file not found: {file_path}")
                continue

            metrics = parse_normals_file(file_path, target_metrics)
            for metric_name, value in metrics.items():
                key = f"{dataset}/{metric_name}"
                method_results[key] = value

        if method_results:
            results[method] = method_results

    # Create and display DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    df.index.name = 'Method'
    # df = df.sort_index(axis=1)

    # print(df)

    # # To percentage
    # df = df * 100
    
    # Rename columns
    def shorten_col(col):
        if "/" in col:
            dataset, metric = col.split("/", 1)
            dataset_short = dataset_shortnames.get(dataset, dataset)
            metric_short = metric_shortnames.get(metric, metric)
            return f"{dataset_short}/{metric_short}"
        return col

    df.columns = [shorten_col(col) for col in df.columns]
    # # print(df)

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
    parser.add_argument("--eval_root", default="./quant/normals/dsine", help="Root directory containing method folders")
    parser.add_argument("--datasets", nargs='+', default=["nyuv2", "scannet", "ibims", "sintel"], help="List of datasets to look for within each method folder")
    parser.add_argument("--target_metrics", nargs='+', default=["mean", "11.25"], help="List of metrics to extract")

    args = parser.parse_args()
    main(args.eval_root, args.methods, args.datasets, args.target_metrics)


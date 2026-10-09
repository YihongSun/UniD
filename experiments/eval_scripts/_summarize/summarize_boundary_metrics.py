import os
import re
import argparse
from pathlib import Path

def parse_f1_score(summary_file):
    """Parse F1 score from boundary_odsF_summary_t0.01.txt file."""
    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            # The F1 score is in the last line, last column
            for line in reversed(lines):
                line = line.strip()
                if line and not line.startswith('|') or 'THRD' in line:
                    continue
                if line.startswith('|'):
                    # Split by | and get the F1 column (last one before final |)
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        f1_score = float(parts[3])
                        return f1_score
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error parsing {summary_file}: {e}")
        return None
    return None

def parse_temporal_consistency(temp_file):
    """Parse temporal consistency from boundary_temp_const_t0.01.txt file."""
    try:
        with open(temp_file, 'r') as f:
            lines = f.readlines()
            # Format: threshold, temporal coverage
            for line in lines:
                if line.strip() and not line.startswith('threshold'):
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        temp_consistency = float(parts[1]) * 100  # Convert to percentage
                        return temp_consistency
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error parsing {temp_file}: {e}")
        return None
    return None

def extract_metrics(experiment_dir):
    """Extract metrics from experiment directory."""
    experiment_path = Path(experiment_dir)
    
    if not experiment_path.exists():
        print(f"Error: Directory {experiment_dir} does not exist!")
        return None
    
    # Define dataset subdirectories and their display names
    datasets = {
        'coco_val': 'COCO F1',
        'lvis_val': 'LVIS F1',
        'ytvis_val': 'YT-VIS F1',
        'mapillary_val': 'Mapillary F1'
    }
    
    results = {}
    
    # Extract F1 scores for each dataset
    for dataset_dir, display_name in datasets.items():
        dataset_path = experiment_path / dataset_dir
        summary_file = dataset_path / 'boundary_odsF_summary_t0.01.txt'
        
        if summary_file.exists():
            f1_score = parse_f1_score(summary_file)
            if f1_score is not None:
                results[display_name] = f1_score
        else:
            print(f"Warning: {summary_file} not found")
            results[display_name] = None
    
    # Extract temporal consistency for YT-VIS
    ytvis_path = experiment_path / 'ytvis_val'
    temp_file = ytvis_path / 'boundary_temp_const_t0.01.txt'
    
    if temp_file.exists():
        temp_consistency = parse_temporal_consistency(temp_file)
        results['YT-VIS Temp'] = temp_consistency
    else:
        print(f"Warning: {temp_file} not found")
        results['YT-VIS Temp'] = None
    
    return results

def print_results_table(results, file_path):
    """Print results in a formatted table."""

    fh = open(file_path, 'w')
    
    # Define column order
    columns = ['COCO F1', 'LVIS F1', 'Mapillary F1', 'YT-VIS F1', 'YT-VIS Temp']
    
    # Print header
    header = "|"
    for col in columns:
        header += f"  {col:^12} |"
    print(header)
    print(header, file=fh)
    
    # Print separator
    separator = "|"
    for _ in columns:
        separator += "-" * 15 + "|"
    print(separator)
    print(separator, file=fh)
    
    # Print values
    row = "|"
    for col in columns:
        value = results.get(col)
        if value is not None:
            row += f"     {value:>6.2f}    |"
        else:
            row += f"      N/A     |"
    print(row)
    print(row, file=fh)
    fh.close()
    print(f"Results saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Extract F1 scores and temporal consistency from experiment results'
    )
    parser.add_argument(
        'experiment_dir',
        type=str,
        help='Path to experiment directory'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='output_path'
    )
    
    args = parser.parse_args()
    
    print(f"Extracting metrics from: {args.experiment_dir}\n")
    
    results = extract_metrics(args.experiment_dir)
    
    if results:
        print_results_table(results, args.output_file)
    else:
        print("Failed to extract metrics")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
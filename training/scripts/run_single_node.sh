#!/bin/bash

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: please provide a YAML config file."
    echo "Usage: $0 <run_script.py> <config_file.yaml> (--overwrite ...)"
    exit 1
fi
run_script=$1
args="${@:2}"

# Get number of GPUs
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")

if [ $num_gpus -gt 1 ]; then
    USE_MULTI_GPU="--multi_gpu"
else
    USE_MULTI_GPU=""
fi

set -x
# Launch run_script
accelerate launch \
  --num_machines 1 \
  $USE_MULTI_GPU \
  --num_processes $num_gpus \
  --mixed_precision no \
  --dynamo_backend no \
  $run_script \
  $args
#!/bin/bash

# Check if config file is provided
if [ -z "$1" ]; then
    echo "Error: please provide a YAML config file."
    echo "Usage: $0 <run_script.py> <config_file.yaml>"
    exit 1
fi
run_script=$1
config_file=$2

# Handle --pods flag
shift  # Remove first argument
shift  # Remove second argument
# Parse node list
PODS=()
if [ "$1" == "--pods" ]; then
    shift  # Remove --pods flag
    # Collect all remaining arguments as node numbers
    PODS=("$@")
else
    # Default: use all PODS from 0 to WORLD_SIZE-1
    for ((i=0; i<WORLD_SIZE; i++)); do
        PODS+=($i)
    done
fi

# Distributed settings
num_nodes=${#PODS[@]}
MASTER_ADDR=${MASTER_ADDR}
num_gpus=$(python -c "import torch; print(torch.cuda.device_count())")
node_rank=${RANK}
echo "NUM_GPUS: $num_gpus, NODES: ${PODS[*]}, NODE_RANK: $node_rank, MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

set -x
# Launch distillation
accelerate launch \
  --multi_gpu \
  --num_processes $((num_nodes * num_gpus)) \
  --num_machines $num_nodes \
  --rdzv_backend c10d \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $node_rank \
  --mixed_precision no \
  --dynamo_backend no \
  $run_script \
  $config_file
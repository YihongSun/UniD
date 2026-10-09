#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${UNID_DATA_ROOT:-data}"
require_local_path() {
    if [ ! -e "$DATA_ROOT/$1" ]; then
        echo "Missing local research data: $DATA_ROOT/$1" >&2
        echo "Obtain this dataset from its upstream source and set UNID_DATA_ROOT to the directory containing the expected layout." >&2
        exit 2
    fi
}
require_local_path "marigold_eval"
# Set checkpoint
if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path_depth> <checkpoint_path_normal> (optional, if different from depth checkpoint)"
    exit 1
fi
checkpoint_path=$1
checkpoint=$(basename "$checkpoint_path")

if [ -z "$2" ]; then
    checkpoint_path_normal=$checkpoint_path
else
    checkpoint_path_normal=$2
fi
checkpoint_normal=$(basename "$checkpoint_path_normal")

# Set text prompt
# Set model type
if [[ "$checkpoint_path" == *"/sd2"* || "$checkpoint_path" == *"/unid"* ]]; then
    model_type="unid"
    text_prompt="video_dense_pixel_prediction"
elif [[ "$checkpoint_path" == *"/dinov"* ]]; then
    model_type="dino"
    text_prompt="SHOULD_BE_IGNORED"
else
    echo "Error: unknown model type in checkpoint path"
    exit 1
fi


set -e
set -x

# infer: scannet
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_scannet_val.yaml \
    --output_dir="quant/ctc/depth/$checkpoint/scannet/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path_normal" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_scannet_val.yaml \
    --output_dir="quant/ctc/normal/$checkpoint_normal/scannet/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj normal

python experiments/eval_scripts/_cross_task/eval_ctc_dn.py \
    --d_pred_dir="quant/ctc/depth/$checkpoint/scannet/prediction" \
    --n_pred_dir="quant/ctc/normal/$checkpoint_normal/scannet/prediction" \
    --split_file="Unid/data_split/scannet/scannet_val_sampled_list_800_1.txt" \
    --output_dir="quant/ctc/ctc_depth_normal/$checkpoint" \
    --num_workers 4

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
require_local_path "hypersim/processed/val"
# Set checkpoint
if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path_albedo> <checkpoint_path_illum> (optional, if different from albedo checkpoint)"
    exit 1
fi
checkpoint_path=$1
checkpoint=$(basename "$checkpoint_path")

if [ -z "$2" ]; then
    checkpoint_path_illum=$checkpoint_path
else
    checkpoint_path_illum=$2
fi
checkpoint_illum=$(basename "$checkpoint_path_illum")

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

# infer: hypersim
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/hypersim" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_hypersim_val.yaml \
    --output_dir="quant/ctc/albedo/$checkpoint/hypersim_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj albedo \
    --save_png

python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path_illum" \
    --base_data_dir="${DATA_ROOT}/hypersim" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_hypersim_val.yaml \
    --output_dir="quant/ctc/illum/$checkpoint_illum/hypersim_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj illum \
    --save_png

python experiments/eval_scripts/_cross_task/eval_ctc_ai.py \
    --a_pred_dir="quant/ctc/albedo/$checkpoint/hypersim_val/prediction" \
    --i_pred_dir="quant/ctc/illum/$checkpoint_illum/hypersim_val/prediction" \
    --data_dir="${DATA_ROOT}/hypersim/processed/val" \
    --split_file="Unid/data_split/hypersim/filename_list_val_filtered.txt" \
    --output_dir="quant/ctc/ctc_albedo_illum/$checkpoint" \
    --num_gpus 4

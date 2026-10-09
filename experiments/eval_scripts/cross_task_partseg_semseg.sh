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
require_local_path "coco/val2014"
require_local_path "coco/annotations/densepose_coco_2014_valminusminival.json"
# Set checkpoint
if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path_partseg> <checkpoint_path_semseg> (optional, if different from partseg checkpoint)"
    exit 1
fi
checkpoint_path=$1
checkpoint=$(basename "$checkpoint_path")

if [ -z "$2" ]; then
    checkpoint_path_semseg=$checkpoint_path
else
    checkpoint_path_semseg=$2
fi
checkpoint_semseg=$(basename "$checkpoint_path_semseg")

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

# infer: coco
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_coco_partseg_val.yaml \
    --output_dir="quant/ctc/partseg-coco/$checkpoint/densepose_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj partseg-coco \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/coco/annotations/inf_dp2dp.npy"

python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path_semseg" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_coco_partseg_val.yaml \
    --output_dir="quant/ctc/semseg-coco/$checkpoint_semseg/densepose_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj semseg-coco \
    --save_argmax \

python experiments/eval_scripts/_cross_task/eval_ctc_ps.py \
    --p_pred_dir="quant/ctc/partseg-coco/$checkpoint/densepose_val/prediction" \
    --s_pred_dir="quant/ctc/semseg-coco/$checkpoint_semseg/densepose_val/prediction" \
    --data_dir="${DATA_ROOT}/coco/val2014" \
    --split_file="Unid/data_split/coco/val2014_partseg.txt" \
    --output_dir="quant/ctc/ctc_partseg_semseg/$checkpoint" \
    --num_workers 4

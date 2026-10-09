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
require_local_path "coco/annotations/inf_dp2dp.npy"
require_local_path "pascal_person"
# Set checkpoint
if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path> <text_prompt>"
    exit 1
fi
checkpoint_path=$1
checkpoint=$(basename "$checkpoint_path")

# Set text prompt
# Set model type
if [[ "$checkpoint_path" == *"/sd2"* || "$checkpoint_path" == *"/unid"* ]]; then
    model_type="unid"
    text_prompt="video_dense_pixel_prediction"
    model_param=$2
elif [[ "$checkpoint_path" == *"/dinov"* ]]; then
    model_type="dino"
    text_prompt="SHOULD_BE_IGNORED"
    model_param="()"
else
    echo "Error: unknown model type in checkpoint path"
    exit 1
fi


set -e
set -x

python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_coco_partseg_val.yaml \
    --output_dir="quant/partseg-coco/$checkpoint/coco_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj partseg-coco \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/coco/annotations/inf_dp2dp.npy"

python experiments/eval_scripts/_partseg/eval_partseg.py \
    --pred_dir="quant/partseg-coco/$checkpoint/coco_val/prediction" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --split="val2014" \
    --output_dir="quant/partseg-coco/$checkpoint/coco_val"

python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/pascal_person" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_pascal_partseg_val.yaml \
    --output_dir="quant/partseg-coco/$checkpoint/pascal_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj partseg-coco \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/pascal_person/shared_dp2pascal.npy"

python experiments/eval_scripts/_partseg/eval_partseg_pascal.py \
    --pred_dir="quant/partseg-coco/$checkpoint/pascal_val/prediction" \
    --base_data_dir="${DATA_ROOT}/pascal_person" \
    --split="val" \
    --output_dir="quant/partseg-coco/$checkpoint/pascal_val"

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/partseg-coco/$checkpoint"

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
require_local_path "coco/val2017"
require_local_path "coco/annotations/panoptic_val2017"
require_local_path "coco/annotations/panoptic_val2017.json"
require_local_path "coco/annotations/lvis_v1_val.json"
require_local_path "YTVIS_val"
require_local_path "mapillary"
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

# Run: COCO 2017val
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_coco_boundary_val.yaml \
    --output_dir="quant/boundary/$checkpoint/coco_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj boundary \

# Eval: coco 2017val
python experiments/eval_scripts/_boundary/eval_boundary.py \
    --pred_dir="quant/boundary/$checkpoint/coco_val/prediction" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --split="val2017" \
    --output_dir="quant/boundary/$checkpoint/coco_val" \

# Eval: lvis val
python experiments/eval_scripts/_boundary/eval_boundary_lvis.py \
    --pred_dir="quant/boundary/$checkpoint/coco_val/prediction" \
    --base_data_dir="${DATA_ROOT}/coco" \
    --split="val" \
    --output_dir="quant/boundary/$checkpoint/lvis_val" \

# Run: ytvis val
bash ./experiments/eval_scripts/_video/run_video.sh "$checkpoint_path" "${DATA_ROOT}/YTVIS_val/JPEGImages" boundary "$model_param" ytvis_val

# Eval: ytvis val
python3 experiments/eval_scripts/_boundary/eval_boundary_ytvis.py \
    --pred_dir="quant/video_boundary/$checkpoint$model_param/ytvis_val/prediction" \
    --output_dir="quant/boundary/$checkpoint/ytvis_val" \
    --num_frames 4

# Run: Mapillary val
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/mapillary" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_mapillary_val.yaml \
    --output_dir="quant/boundary/$checkpoint/mapillary_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj boundary \
    --processing_res 1024

# Eval: Mapillary val
python experiments/eval_scripts/_boundary/eval_boundary_mapillary.py \
    --pred_dir="quant/boundary/$checkpoint/mapillary_val/prediction" \
    --base_data_dir="${DATA_ROOT}/mapillary" \
    --split="validation" \
    --output_dir="quant/boundary/$checkpoint/mapillary_val" \
    --eval_res 1024


python experiments/eval_scripts/_summarize/summarize_boundary_metrics.py \
    "quant/boundary/$checkpoint" \
    "quant/boundary/$checkpoint/summary_metrics.txt"

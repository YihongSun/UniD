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
require_local_path "DMS_v1"
require_local_path "ade20k"
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
    --base_data_dir="${DATA_ROOT}/DMS_v1" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_dms_matseg_val.yaml \
    --output_dir="quant/matseg/$checkpoint/dms_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj matseg \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/DMS_v1/inf_dms2dms.npy"

python experiments/eval_scripts/_matseg/eval_matseg.py \
    --pred_dir="quant/matseg/$checkpoint/dms_val/prediction" \
    --base_data_dir="${DATA_ROOT}/DMS_v1" \
    --split="validation" \
    --output_dir="quant/matseg/$checkpoint/dms_val"

## == ade20k Eval
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/ade20k" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_ade20k_val.yaml \
    --output_dir="quant/matseg/$checkpoint/ade20k_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj matseg \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/DMS_v1/inf_dms2dms.npy"

python experiments/eval_scripts/_matseg/eval_matseg_ade20k.py \
    --pred_dir="quant/matseg/$checkpoint/ade20k_val/prediction" \
    --base_data_dir="${DATA_ROOT}/ade20k" \
    --split="validation" \
    --output_dir="quant/matseg/$checkpoint/ade20k_val"

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/matseg/$checkpoint"

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
require_local_path "VIPSeg_val"
require_local_path "cityscapes"
require_local_path "ade20k"
# Set checkpoint
if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path> <text_prompt>"
    exit 1
fi
CHECKPOINT_PATH=$1
checkpoint=$(basename "$CHECKPOINT_PATH")
MODEL_PARAM=$2



# Set text prompt
# Set model type
if [[ "$CHECKPOINT_PATH" == *"/sd2"* || "$CHECKPOINT_PATH" == *"/unid"* ]]; then
    model_type="unid"
    text_prompt="video_dense_pixel_prediction"
elif [[ "$CHECKPOINT_PATH" == *"/dinov"* ]]; then
    model_type="dino"
    text_prompt="SHOULD_BE_IGNORED"
    MODEL_PARAM="()"
else
    echo "Error: unknown model type in checkpoint path"
    exit 1
fi

## == VIPSeg Video Eval
bash ./experiments/eval_scripts/_video/run_video.sh \
    "$CHECKPOINT_PATH" \
    ${DATA_ROOT}/VIPSeg_val/imgs \
    semseg-vipseg \
    "$MODEL_PARAM" \
    vipseg_val \

python ./experiments/eval_scripts/_semseg/eval_semseg_video.py --dataset_path ${DATA_ROOT}/VIPSeg_val/ --model "$checkpoint$MODEL_PARAM"

## == Cityscapes Eval
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$CHECKPOINT_PATH" \
    --base_data_dir="${DATA_ROOT}/cityscapes" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_cityscapes_semseg_val.yaml \
    --output_dir="quant/semseg/$checkpoint$MODEL_PARAM/cityscapes_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj semseg-cityscapes \
    --save_argmax \

python experiments/eval_scripts/_semseg/eval_semseg_cityscapes.py \
    --pred_dir="quant/semseg/$checkpoint$MODEL_PARAM/cityscapes_val/prediction" \
    --base_data_dir="${DATA_ROOT}/cityscapes" \
    --split="val" \
    --output_dir="quant/semseg/$checkpoint$MODEL_PARAM/cityscapes_val"

## == ade20k Eval
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$CHECKPOINT_PATH" \
    --base_data_dir="${DATA_ROOT}/ade20k" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_ade20k_val.yaml \
    --output_dir="quant/semseg/$checkpoint$MODEL_PARAM/ade20k_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj semseg-coco \
    --save_argmax \
    --label_mapping "${DATA_ROOT}/ade20k/coco_annotations/shared_coco2uni.npy"

python experiments/eval_scripts/_semseg/eval_semseg_ade20k.py \
    --pred_dir="quant/semseg/$checkpoint$MODEL_PARAM/ade20k_val/prediction" \
    --base_data_dir="${DATA_ROOT}/ade20k" \
    --split="validation" \
    --output_dir="quant/semseg/$checkpoint$MODEL_PARAM/ade20k_val"

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/semseg/$checkpoint$MODEL_PARAM"

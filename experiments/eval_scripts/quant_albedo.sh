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
require_local_path "iiw"
require_local_path "interiornet"
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

# infer: hypersim
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/hypersim" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_hypersim_val.yaml \
    --output_dir="quant/albedo/$checkpoint/hypersim_val/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj albedo \
    --save_png

# eval: hypersim
python experiments/eval_scripts/_recon/eval_reconstruction.py \
    --pred_dir="quant/albedo/$checkpoint/hypersim_val/prediction" \
    --gt_dir="${DATA_ROOT}/hypersim/processed/val" \
    --split_file="Unid/data_split/hypersim/filename_list_val_filtered.txt" \
    --output_dir="quant/albedo/$checkpoint/hypersim_val" \
    --task albedo

# infer: iiw
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/iiw" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_iiw_test.yaml \
    --output_dir="quant/albedo/$checkpoint/iiw_test/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj albedo \
    --save_png

# eval: iiw
python experiments/eval_scripts/_albedo/eval_albedo_iiw.py \
    --pred_dir="quant/albedo/$checkpoint/iiw_test/prediction" \
    --gt_dir="${DATA_ROOT}/iiw" \
    --split_file="Unid/data_split/iiw/test.txt" \
    --output_dir="quant/albedo/$checkpoint/iiw_test"

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/albedo/$checkpoint"


# Eval Video Albedo
bash ./experiments/eval_scripts/_video/run_video.sh \
    "$checkpoint_path" \
    ${DATA_ROOT}/interiornet \
    albedo \
    "$model_param" \
    interiornet_val \
    /cam0 \
    "--stride 1 --early_break 100"

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_albedo/$checkpoint$model_param/interiornet_val/prediction \
    --modality albedo \
    --vc_len 4 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_albedo/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_albedo/$checkpoint$model_param/interiornet_val/prediction \
    --modality albedo \
    --vc_len 8 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_albedo/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_albedo/$checkpoint$model_param/interiornet_val/prediction \
    --modality albedo \
    --vc_len 16 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_albedo/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/video_albedo/$checkpoint$model_param"

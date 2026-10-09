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
require_local_path "dsine_eval"
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

# Define the path to the temporary args file
current_time=$(date "+%Y-%m-%d-%H-%M-%S")
ARGS_FILE="tmp_normals_args_$current_time.txt"

# Write the arguments to the file
cat > "$ARGS_FILE" <<EOF
--exp_name $checkpoint
--ckpt_path $checkpoint_path
--model_type $model_type
--eval_data all
--processing_res 0
--seed 1234
EOF

# Run the test
PYTHONPATH="experiments/eval_scripts/_normal:$PYTHONPATH" python -m DSINE.projects.dsine.test \
    $ARGS_FILE \
    --mode benchmark \
    --log_root quant/normals \
    --text_prompt "$text_prompt"

# Remove args file
rm "$ARGS_FILE"

# Summarize metrics
echo ""
echo "Summary:"
python experiments/eval_scripts/_summarize/summarize_normals_metrics.py --methods "$checkpoint" | tee quant/normals/dsine/$checkpoint/summary.txt

# Eval Video Normal
bash ./experiments/eval_scripts/_video/run_video.sh \
    "$checkpoint_path" \
    ${DATA_ROOT}/interiornet \
    normal \
    "$model_param" \
    interiornet_val \
    /cam0 \
    "--stride 1 --early_break 100"

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_normal/$checkpoint$model_param/interiornet_val/prediction \
    --modality normal \
    --vc_len 4 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_normal/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_normal/$checkpoint$model_param/interiornet_val/prediction \
    --modality normal \
    --vc_len 8 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_normal/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

python3 experiments/eval_scripts/_video/eval_pred_vc.py \
    --pred_dir quant/video_normal/$checkpoint$model_param/interiornet_val/prediction \
    --modality normal \
    --vc_len 16 \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_normal/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

python3 experiments/eval_scripts/_normal/eval_pred_perf.py \
    --pred_dir quant/video_normal/$checkpoint$model_param/interiornet_val/prediction \
    --modality normal \
    --data_dir ${DATA_ROOT}/interiornet \
    --output_dir quant/video_normal/$checkpoint$model_param/interiornet_val \
    --max_length 100 \
    --stride 1 \
    --num_workers 2

# summarize metrics
python3 experiments/eval_scripts/_summarize/summarize_metrics.py \
    --eval_root "quant/video_normal/$checkpoint$model_param"

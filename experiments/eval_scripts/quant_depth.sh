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
require_local_path "video_depth_data"
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
    alignment="least_square"
elif [[ "$checkpoint_path" == *"/dinov"* ]]; then
    model_type="dino"
    text_prompt="SHOULD_BE_IGNORED"
    model_param="()"
    alignment="least_square"
else
    echo "Error: unknown model type in checkpoint path"
    exit 1
fi


set -e
set -x

# infer: nyu
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_nyu_test.yaml \
    --output_dir="quant/depth/$checkpoint/nyu_test/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

# eval: nyu
python experiments/eval_scripts/_depth/eval_depth_image.py \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --dataset_config Unid/config/dataset/data_nyu_test.yaml \
    --alignment $alignment \
    --prediction_dir="quant/depth/$checkpoint/nyu_test/prediction" \
    --output_dir="quant/depth/$checkpoint/nyu_test"

# infer: kitti
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_kitti_eigen_test.yaml \
    --output_dir="quant/depth/$checkpoint/kitti_eigen_test/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

# eval: kitti
python experiments/eval_scripts/_depth/eval_depth_image.py \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --dataset_config Unid/config/dataset/data_kitti_eigen_test.yaml \
    --alignment $alignment \
    --prediction_dir "quant/depth/$checkpoint/kitti_eigen_test/prediction" \
    --output_dir "quant/depth/$checkpoint/kitti_eigen_test"

# infer: eth3d
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 756 \
    --dataset_config Unid/config/dataset/data_eth3d.yaml \
    --output_dir="quant/depth/$checkpoint/eth3d/prediction" \
    --resample_method bilinear \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

# eval: eth3d
python experiments/eval_scripts/_depth/eval_depth_image.py \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --dataset_config Unid/config/dataset/data_eth3d.yaml \
    --alignment $alignment \
    --prediction_dir="quant/depth/$checkpoint/eth3d/prediction" \
    --output_dir="quant/depth/$checkpoint/eth3d" \
    --alignment_max_res 1024

# infer: scannet
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 0 \
    --dataset_config Unid/config/dataset/data_scannet_val.yaml \
    --output_dir="quant/depth/$checkpoint/scannet/prediction" \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

# eval: scannet
python experiments/eval_scripts/_depth/eval_depth_image.py \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --dataset_config Unid/config/dataset/data_scannet_val.yaml \
    --alignment $alignment \
    --prediction_dir="quant/depth/$checkpoint/scannet/prediction" \
    --output_dir="quant/depth/$checkpoint/scannet"

# infer: diode
python Unid/infer_image.py \
    --seed 1234 \
    --checkpoint="$checkpoint_path" \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --processing_res 640 \
    --dataset_config Unid/config/dataset/data_diode_all.yaml \
    --output_dir="quant/depth/$checkpoint/diode/prediction" \
    --resample_method bilinear \
    --model_type $model_type \
    --text_prompt "$text_prompt" \
    --proj depth

# eval: diode
python experiments/eval_scripts/_depth/eval_depth_image.py \
    --base_data_dir="${DATA_ROOT}/marigold_eval" \
    --dataset_config Unid/config/dataset/data_diode_all.yaml \
    --alignment $alignment \
    --prediction_dir "quant/depth/$checkpoint/diode/prediction" \
    --output_dir "quant/depth/$checkpoint/diode"

# Summarize metrics
echo ""
echo "Summary:"
python experiments/eval_scripts/_summarize/summarize_depth_metrics.py --methods "$checkpoint" --alignment $alignment | tee quant/depth/$checkpoint/summary.txt


# Video Depth Eval
bash ./experiments/eval_scripts/_video/run_video.sh \
    "$checkpoint_path" \
    "${DATA_ROOT}/video_depth_data/scannet" \
    depth \
    "$model_param" \
    scannet_test \
    "/color" \
    "--store_np --stride 3 --early_break 90"

python3 experiments/eval_scripts/_depth/eval_depth_video.py \
    --infer_path quant/video_depth/$checkpoint$model_param/scannet_test/ \
    --benchmark_path "${DATA_ROOT}/video_depth_data/" \
    --datasets scannet \
    --load_depth \
    --alignment $alignment

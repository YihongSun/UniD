#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <checkpoint_path> <video_directory> <proj> <model_params> <eval_name> <custom_pattern> <optional: flags>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: video root directory is not provided"
    echo "Usage: $0 <checkpoint_path> <video_directory> <proj> <model_params> <eval_name> <custom_pattern> <optional: flags>"
    exit 1
fi

if [ -z "$3" ]; then
    echo "Error: proj is not provided"
    echo "Usage: $0 <checkpoint_path> <video_directory> <proj> <model_params> <eval_name> <custom_pattern> <optional: flags>"
    exit 1
fi

if [ -z "$4" ]; then
    echo "Error: model_params is not provided"
    echo "Usage: $0 <checkpoint_path> <video_directory> <proj> <model_params> <eval_name> <custom_pattern> <optional: flags>"
    exit 1
fi

if [ -z "$5" ]; then
    echo "Error: eval_name is not provided"
    echo "Usage: $0 <checkpoint_path> <video_directory> <proj> <model_params> <eval_name> <custom_pattern> <optional: flags>"
    exit 1
fi

CHECKPOINT_PATH="$1"
CKPT_NAME=$(basename "$CHECKPOINT_PATH")
VIDEO_DIR_PATH="$2"
PROJ="$3"
MODEL_PARAMS="$4"
eval_name="$5"
CUSTOM_PATTERN="${6:-}"  # Default to empty string if not provided
flags="${7:-}"  # Default to empty string if not provided
OUTPUT_FRAME_ROOT="quant/video_${PROJ}/${CKPT_NAME}${MODEL_PARAMS}/${eval_name}/prediction"

# Set text prompt
# Set model type
if [[ "$CHECKPOINT_PATH" == *"/sd2"* || "$CHECKPOINT_PATH" == *"/unid"* ]]; then
    model_type="unid"
    text_prompt="video_dense_pixel_prediction"
elif [[ "$CHECKPOINT_PATH" == *"/dinov"* ]]; then
    model_type="dino"
    text_prompt="SHOULD_BE_IGNORED"
else
    echo "Error: unknown model type in checkpoint path"
    exit 1
fi

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using checkpoint path: $CHECKPOINT_PATH"
echo "Using video directory: $VIDEO_DIR_PATH"
echo "Number of GPUs available: $NUM_GPUS"
echo "Using model params: $MODEL_PARAMS"
echo "Model type: $model_type"
echo "Text prompt: $text_prompt"

video_list=""
video_count=0
for VIDEO_PATH in "$VIDEO_DIR_PATH"/*; do
    if [ -d "$VIDEO_PATH" ]; then
        video_list="$video_list $VIDEO_PATH"
        video_count=$((video_count + 1))
    fi
done

echo "Found $video_count video files"

for gpu in $(seq 0 $((NUM_GPUS-1))); do
    (
        # Process every NUM_GPUS-th file starting from GPU ID
        current_index=0
        for VIDEO_PATH in $video_list; do
            if [ $((current_index % NUM_GPUS)) -eq $gpu ]; then
                echo "GPU $gpu processing: $VIDEO_PATH"
                VID_NAME=$(basename "$VIDEO_PATH")
                CUDA_VISIBLE_DEVICES=$gpu python Unid/run.py \
                    --checkpoint="$CHECKPOINT_PATH" \
                    --projs="$PROJ" \
                    --input_rgb="$VIDEO_PATH$CUSTOM_PATTERN" \
                    --output_dir="$OUTPUT_FRAME_ROOT/"$(basename "$VIDEO_PATH") \
                    --text_prompt=$text_prompt \
                    --model_params="$MODEL_PARAMS" \
                    --model_type=$model_type \
                    $flags
            fi
            current_index=$((current_index + 1))
        done
    ) &
done
wait
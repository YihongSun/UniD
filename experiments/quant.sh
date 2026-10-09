#!/usr/bin/env bash

# Usage:
# bash ./experiments/quant.sh M model-finetuned/sd2_uni_M_MemAttnTemp\(dec\)_T4_80000_v1/checkpoint-50000

# Set eval function - check task parameter
if [ -z "$1" ]; then
    echo "Error: task letter code is not provided"
    echo "Usage: $0 <task> <checkpoint_path> OPTIONAL<video_inf_param>"
    exit 1
fi

# Set checkpoint - check checkpoint parameter
if [ -z "$2" ]; then
    echo "Error: checkpoint path is not provided"
    echo "Usage: $0 <task> <checkpoint_path>"
    exit 1
fi

task=$1
checkpoint_path=$2
if [ "$#" -gt 2 ]; then
    video_inf_param=$3
else
    video_inf_param="(len=16,st=1,pop_loc=0)"
fi

if [[ "$task" == *"D"* ]]; then
    bash ./experiments/eval_scripts/quant_depth.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"N"* ]]; then
    bash ./experiments/eval_scripts/quant_normal.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"A"* ]]; then
    bash ./experiments/eval_scripts/quant_albedo.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"I"* ]]; then
    bash ./experiments/eval_scripts/quant_illum.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"P"* ]]; then
    bash ./experiments/eval_scripts/quant_partseg.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"M"* ]]; then
    bash ./experiments/eval_scripts/quant_matseg.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"B"* ]]; then
    bash ./experiments/eval_scripts/quant_boundary.sh "$checkpoint_path" $video_inf_param
fi
if [[ "$task" == *"S"* ]]; then
    bash ./experiments/eval_scripts/quant_semseg.sh "$checkpoint_path" $video_inf_param
fi
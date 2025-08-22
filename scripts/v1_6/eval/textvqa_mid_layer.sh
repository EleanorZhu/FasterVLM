#!/bin/bash

# FasterVLM: Evaluation script for mid-layer concentrated pruning (方案D)
# Usage: ./textvqa_mid_layer.sh [mid_layer_ratios] [mid_layer_layers]
# Default mid-layer ratios: 0.1,0.5,0.9 (10%, 50%, 90% cumulative pruning)
# Default mid-layer layers: 6,12,18 (no pruning at layer 24)

CKPT="llava-v1.6-vicuna-7b"
METHOD="fastervlm_mid_layer"
MID_LAYER_RATIOS=${1:-"0.1,0.5,0.9"}  # Default mid-layer ratios
MID_LAYER_LAYERS=${2:-"6,12,18"}  # Default mid-layer layers

# Create a parameter name from the mid-layer settings
PARAM="mid_$(echo $MID_LAYER_RATIOS | tr ',' '_')_L$(echo $MID_LAYER_LAYERS | tr ',' '_')"

echo "Running TextVQA evaluation with mid-layer concentrated pruning"
echo "Mid-layer ratios: ${MID_LAYER_RATIOS}"
echo "Mid-layer layers: ${MID_LAYER_LAYERS}"
echo "Checkpoint: ${CKPT}"
echo "Method: ${METHOD}"
echo "Parameter: ${PARAM}"

# Create output directory
mkdir -p ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --enable-mid-layer-pruning \
    --mid-layer-ratios ${MID_LAYER_RATIOS} \
    --mid-layer-layers ${MID_LAYER_LAYERS} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl

echo "Evaluation completed. Results saved to:"
echo "./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl"
#!/bin/bash

# FasterVLM: Evaluation script for progressive pruning (方案C)
# Usage: ./textvqa_progressive.sh [progressive_ratios] [progressive_layers]
# Default progressive ratios: 0.2,0.5,0.8,0.9 (20%, 50%, 80%, 90% cumulative pruning)
# Default progressive layers: 6,12,18,24

CKPT="llava-v1.6-vicuna-7b"
METHOD="fastervlm_progressive"
PROGRESSIVE_RATIOS=${1:-"0.2,0.5,0.8,0.9"}  # Default progressive ratios
PROGRESSIVE_LAYERS=${2:-"6,12,18,24"}  # Default progressive layers

# Create a parameter name from the progressive settings
PARAM="prog_$(echo $PROGRESSIVE_RATIOS | tr ',' '_')_L$(echo $PROGRESSIVE_LAYERS | tr ',' '_')"

echo "Running TextVQA evaluation with progressive pruning"
echo "Progressive ratios: ${PROGRESSIVE_RATIOS}"
echo "Progressive layers: ${PROGRESSIVE_LAYERS}"
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
    --enable-progressive-pruning \
    --progressive-ratios ${PROGRESSIVE_RATIOS} \
    --progressive-layers ${PROGRESSIVE_LAYERS} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl

echo "Evaluation completed. Results saved to:"
echo "./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl"
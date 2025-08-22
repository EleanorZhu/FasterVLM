#!/bin/bash

# FasterVLM: Evaluation script for layer-18 pruning (方案B)
# Usage: ./textvqa_layer18.sh [pruning_ratio]
# Default pruning ratio is 0.9 (90% pruning)

CKPT="llava-v1.6-vicuna-7b"
METHOD="fastervlm_layer18"
PRUNING_RATIO=${1:-0.9}  # Default to 0.9 if no argument provided
PARAM="layer18_pr_${PRUNING_RATIO}"

echo "Running TextVQA evaluation with layer-18 pruning"
echo "Pruning ratio: ${PRUNING_RATIO}"
echo "Checkpoint: ${CKPT}"
echo "Method: ${METHOD}"

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --enable-layer18-pruning \
    --pruning-ratio ${PRUNING_RATIO} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl

echo "Evaluation completed. Results saved to:"
echo "./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl"
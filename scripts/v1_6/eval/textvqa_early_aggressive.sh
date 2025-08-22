#!/bin/bash

# FasterVLM: Evaluation script for early aggressive pruning (方案E - Pressure Test)
# Usage: ./textvqa_early_aggressive.sh [early_aggressive_ratios] [early_aggressive_layers]
# Default early aggressive ratios: 0.7,0.9 (70%, 90% cumulative pruning)
# Default early aggressive layers: 6,12 (no pruning at layer 18, 24)

CKPT="llava-v1.6-vicuna-7b"
METHOD="fastervlm_early_aggressive"
EARLY_AGGRESSIVE_RATIOS=${1:-"0.7,0.9"}  # Default early aggressive ratios - PRESSURE TEST
EARLY_AGGRESSIVE_LAYERS=${2:-"6,12"}  # Default early aggressive layers

# Create a parameter name from the early aggressive settings
PARAM="early_$(echo $EARLY_AGGRESSIVE_RATIOS | tr ',' '_')_L$(echo $EARLY_AGGRESSIVE_LAYERS | tr ',' '_')"

echo "=============================================="
echo "FasterVLM Early Aggressive Pruning (方案E)"
echo "             PRESSURE TEST"
echo "=============================================="
echo "Early aggressive ratios: ${EARLY_AGGRESSIVE_RATIOS}"
echo "Early aggressive layers: ${EARLY_AGGRESSIVE_LAYERS}"
echo "Checkpoint: ${CKPT}"
echo "Method: ${METHOD}"
echo "Parameter: ${PARAM}"
echo ""
echo "WARNING: This is a pressure test designed to"
echo "validate the hypothesis that shallow layers"
echo "lack sufficient semantic understanding for"
echo "reliable pruning decisions."
echo "Performance is expected to be significantly"
echo "degraded compared to other methods."
echo "=============================================="

# Create output directory
mkdir -p ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --enable-early-aggressive-pruning \
    --early-aggressive-ratios ${EARLY_AGGRESSIVE_RATIOS} \
    --early-aggressive-layers ${EARLY_AGGRESSIVE_LAYERS} \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl

echo ""
echo "=============================================="
echo "Early Aggressive Pruning Evaluation Completed"
echo "=============================================="
echo "Results saved to:"
echo "./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl"
echo ""
echo "If accuracy is significantly lower than other"
echo "methods, this validates the hypothesis that"
echo "deeper layers are necessary for reliable"
echo "pruning decisions."
echo "=============================================="
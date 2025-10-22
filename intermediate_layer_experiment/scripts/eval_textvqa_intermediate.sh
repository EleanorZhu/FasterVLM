#!/bin/bash

# Evaluation script for TextVQA with intermediate layer feedback
# Usage: bash eval_textvqa_intermediate.sh <layer_idx> [visual_token_num] [use_feedback]
#
# Example:
#   bash eval_textvqa_intermediate.sh 3 576 true   # Use intermediate layer feedback
#   bash eval_textvqa_intermediate.sh 3 576 false  # Only use projector output
#   bash eval_textvqa_intermediate.sh 5            # Default: use feedback

set -e

# Set Python path - use micromamba environment if available
if [ -f "$HOME/micromamba/envs/fastervlm/bin/python" ]; then
    PYTHON="$HOME/micromamba/envs/fastervlm/bin/python"
else
    PYTHON="python"
fi

# Configuration
CKPT="llava-v1.5-7b"
LAYER_IDX=${1:-3}       # Default to layer 3
TOKEN=${2:-576}         # Default to 576 tokens (no pruning)
USE_FEEDBACK=${3:-true} # Default to true (use intermediate feedback)

# Set method name based on feedback flag
if [ "$USE_FEEDBACK" = "true" ]; then
    METHOD="intermediate_layer"
    PARAM="layer${LAYER_IDX}_n${TOKEN}"
else
    METHOD="projector_only"
    PARAM="n${TOKEN}"
fi

# Paths
MODEL_PATH="/home/qingchan/project/FasterVLM/checkpoints/${CKPT}"
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
ANSWERS_FILE="./intermediate_layer_experiment/results/textvqa/${CKPT}/${METHOD}/${PARAM}.jsonl"

echo "=========================================="
if [ "$USE_FEEDBACK" = "true" ]; then
    echo "TextVQA Evaluation with Intermediate Layer Feedback"
else
    echo "TextVQA Evaluation with Projector Output Only"
fi
echo "=========================================="
echo "Model: ${CKPT}"
if [ "$USE_FEEDBACK" = "true" ]; then
    echo "Intermediate Layer: ${LAYER_IDX}"
else
    echo "Mode: Projector output only (no intermediate layer)"
fi
echo "Visual Tokens: ${TOKEN}"
echo "Results will be saved to: ${ANSWERS_FILE}"
echo "=========================================="

# Create output directory
mkdir -p $(dirname ${ANSWERS_FILE})

# Build command with optional --use-intermediate-feedback flag
FEEDBACK_FLAG=""
if [ "$USE_FEEDBACK" = "true" ]; then
    FEEDBACK_FLAG="--use-intermediate-feedback"
fi

# Run inference
echo ""
if [ "$USE_FEEDBACK" = "true" ]; then
    echo "Step 1: Running inference with intermediate layer feedback..."
else
    echo "Step 1: Running inference with projector output only..."
fi
CUDA_VISIBLE_DEVICES=0 ${PYTHON} -W ignore -u intermediate_layer_experiment/scripts/model_vqa_intermediate_layer.py \
    --model-path ${MODEL_PATH} \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ${ANSWERS_FILE} \
    --intermediate-layer-idx ${LAYER_IDX} \
    --visual-token-num ${TOKEN} \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    ${FEEDBACK_FLAG}

# Evaluate results
echo ""
echo "Step 2: Evaluating results..."
${PYTHON} -m llava.eval.eval_textvqa \
    --annotation-file ${ANNOTATION_FILE} \
    --result-file ${ANSWERS_FILE}

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${ANSWERS_FILE}"
echo "=========================================="


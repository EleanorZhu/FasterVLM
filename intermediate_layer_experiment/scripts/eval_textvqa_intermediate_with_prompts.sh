#!/bin/bash

#
# Evaluation script for TextVQA with intermediate layer and prompt tuning
#
# Usage:
#   bash intermediate_layer_experiment/scripts/eval_textvqa_intermediate_with_prompts.sh [LAYER_IDX] [VISUAL_TOKEN_NUM] [PROMPT_TOKENS_PATH]
#
# Examples:
#   # Evaluate with trained prompt tokens
#   bash intermediate_layer_experiment/scripts/eval_textvqa_intermediate_with_prompts.sh 3 576 ./intermediate_layer_experiment/checkpoints/prompt_tokens_layer3_n20_fixed.pt
#
#   # Evaluate without prompt tokens (baseline)
#   bash intermediate_layer_experiment/scripts/eval_textvqa_intermediate_with_prompts.sh 3 576
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Python executable
PYTHON=${PYTHON:-python}

# Configuration
MODEL_PATH="/home/qingchan/project/FasterVLM/checkpoints/llava-v1.5-7b"
LAYER_IDX=${1:-3}                    # Default to layer 3
VISUAL_TOKEN_NUM=${2:-576}           # Default to 576 (no pruning)
PROMPT_TOKENS_PATH=${3:-}            # Optional prompt tokens path

# Data paths
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"

# Output directory
if [ -n "$PROMPT_TOKENS_PATH" ]; then
    # Extract num_prompt_tokens from filename (e.g., prompt_tokens_layer3_n20.pt or prompt_tokens_layer3_n20_fixed.pt -> 20)
    FILENAME=$(basename "$PROMPT_TOKENS_PATH")
    NUM_PROMPT_TOKENS=$(echo "$FILENAME" | sed -n 's/.*_n\([0-9]\+\).*/\1/p')
    if [ -z "$NUM_PROMPT_TOKENS" ]; then
        NUM_PROMPT_TOKENS=10  # Default if can't parse
    fi
    OUTPUT_DIR="./intermediate_layer_experiment/results/textvqa_layer${LAYER_IDX}_prompts${NUM_PROMPT_TOKENS}"
else
    OUTPUT_DIR="./intermediate_layer_experiment/results/textvqa_layer${LAYER_IDX}_no_prompts"
fi

ANSWERS_FILE="${OUTPUT_DIR}/answers.jsonl"

# Evaluation parameters
CONV_MODE="vicuna_v1"
TEMPERATURE=0.0
NUM_BEAMS=1
MAX_NEW_TOKENS=1024

echo "=========================================="
echo "TextVQA Evaluation Configuration"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Intermediate Layer: $LAYER_IDX"
echo "Visual Token Num: $VISUAL_TOKEN_NUM"
if [ -n "$PROMPT_TOKENS_PATH" ]; then
    echo "Prompt Tokens: $PROMPT_TOKENS_PATH"
    echo "Number of Prompt Tokens: $NUM_PROMPT_TOKENS"
else
    echo "Prompt Tokens: None (baseline)"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build prompt tokens arguments
PROMPT_ARGS=""
if [ -n "$PROMPT_TOKENS_PATH" ]; then
    PROMPT_ARGS="--prompt-tokens-path $PROMPT_TOKENS_PATH --num-prompt-tokens $NUM_PROMPT_TOKENS"
fi

# Run evaluation
echo "Step 1: Running inference..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u intermediate_layer_experiment/scripts/model_vqa_intermediate_layer.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$ANSWERS_FILE" \
    --conv-mode "$CONV_MODE" \
    --temperature $TEMPERATURE \
    --num_beams $NUM_BEAMS \
    --max_new_tokens $MAX_NEW_TOKENS \
    --intermediate-layer-idx $LAYER_IDX \
    --visual-token-num $VISUAL_TOKEN_NUM \
    --use-intermediate-feedback \
    $PROMPT_ARGS

# Compute accuracy
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
echo ""
echo "Step 2: Computing accuracy..."
$PYTHON -m llava.eval.eval_textvqa \
    --annotation-file "$ANNOTATION_FILE" \
    --result-file "$ANSWERS_FILE"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $ANSWERS_FILE"
echo "=========================================="


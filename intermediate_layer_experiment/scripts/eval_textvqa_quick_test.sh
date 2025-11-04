#!/bin/bash

#
# Quick test script for TextVQA evaluation (10 samples only)
#
# Usage:
#   bash intermediate_layer_experiment/scripts/eval_textvqa_quick_test.sh [LAYER_IDX] [VISUAL_TOKEN_NUM] [PROMPT_TOKENS_PATH]
#
# Examples:
#   # Test with trained prompt tokens
#   bash intermediate_layer_experiment/scripts/eval_textvqa_quick_test.sh 3 576 ./intermediate_layer_experiment/checkpoints/prompt_tokens_layer3_n20_fixed.pt
#
#   # Test without prompt tokens (baseline)
#   bash intermediate_layer_experiment/scripts/eval_textvqa_quick_test.sh 3 576
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Python executable
PYTHON=${PYTHON:-/home/qingchan/micromamba/envs/fastervlm/bin/python}

# Configuration
MODEL_PATH="/home/qingchan/project/FasterVLM/checkpoints/llava-v1.5-7b"
LAYER_IDX=${1:-3}                    # Default to layer 3
VISUAL_TOKEN_NUM=${2:-576}           # Default to 576 (no pruning)
PROMPT_TOKENS_PATH=${3:-}            # Optional prompt tokens path

# Data paths
QUESTION_FILE_FULL="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
QUESTION_FILE_TEMP="./tmp/samples.jsonl"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"

# Create temporary file with only 10 samples
head -10 "$QUESTION_FILE_FULL" > "$QUESTION_FILE_TEMP"

# Output directory
if [ -n "$PROMPT_TOKENS_PATH" ]; then
    # Extract num_prompt_tokens from filename (e.g., prompt_tokens_layer3_n20_fixed.pt -> 20)
    FILENAME=$(basename "$PROMPT_TOKENS_PATH")
    NUM_PROMPT_TOKENS=$(echo "$FILENAME" | sed -n 's/.*_n\([0-9]\+\).*/\1/p')
    if [ -z "$NUM_PROMPT_TOKENS" ]; then
        NUM_PROMPT_TOKENS=10  # Default if can't parse
    fi
    OUTPUT_DIR="./intermediate_layer_experiment/results/textvqa_layer${LAYER_IDX}_prompts${NUM_PROMPT_TOKENS}_test10"
else
    OUTPUT_DIR="./intermediate_layer_experiment/results/textvqa_layer${LAYER_IDX}_no_prompts_test10"
fi

# Print configuration
echo "=========================================="
echo "TextVQA Quick Test (10 samples)"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Intermediate Layer: $LAYER_IDX"
echo "Visual Token Num: $VISUAL_TOKEN_NUM"
if [ -n "$PROMPT_TOKENS_PATH" ]; then
    echo "Prompt Tokens: $PROMPT_TOKENS_PATH"
    echo "Number of Prompt Tokens: $NUM_PROMPT_TOKENS"
else
    echo "Prompt Tokens: None (baseline test)"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Step 1: Run inference on 10 samples
echo "Step 1: Running inference on 10 samples..."
$PYTHON intermediate_layer_experiment/scripts/model_vqa_intermediate_layer.py \
    --model-path "$MODEL_PATH" \
    --question-file "$QUESTION_FILE_TEMP" \
    --image-folder "$IMAGE_FOLDER" \
    --answers-file "$OUTPUT_DIR/answers.jsonl" \
    --intermediate-layer-idx "$LAYER_IDX" \
    --visual-token-num "$VISUAL_TOKEN_NUM" \
    --max_new_tokens 1024 \
    --use-intermediate-feedback \
    ${PROMPT_TOKENS_PATH:+--prompt-tokens-path "$PROMPT_TOKENS_PATH"}

# # Clean up temporary file
# rm -f "$QUESTION_FILE_TEMP"

echo ""
echo "=========================================="
echo "Quick test complete!"
echo "Results saved to: $OUTPUT_DIR/answers.jsonl"
echo "=========================================="
echo ""

# Show sample outputs
echo "Sample outputs:"
if [ -f "$OUTPUT_DIR/answers.jsonl" ]; then
    head -3 "$OUTPUT_DIR/answers.jsonl" | $PYTHON -m json.tool 2>/dev/null || head -3 "$OUTPUT_DIR/answers.jsonl"
else
    echo "No output file found!"
fi


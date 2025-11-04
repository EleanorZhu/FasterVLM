#!/bin/bash

#
# Training script for prompt tuning on intermediate layer embeddings (FIXED VERSION)
#
# FIXED: Ensures train/inference consistency by:
# 1. Extracting intermediate layer from question only (no answer)
# 2. Appending answer embeddings after intermediate extraction
# 3. This makes intermediate_seq identical between training and inference
#
# Usage:
#   bash intermediate_layer_experiment/scripts/train_prompt_tuning_fixed.sh [LAYER_IDX] [NUM_PROMPT_TOKENS] [MAX_SAMPLES] [VISUAL_TOKEN_NUM]
#   - VISUAL_TOKEN_NUM (optional): number of vision tokens to keep (576 = no pruning for ViT-L/14@336; try 256/144 to prune)
#
# Examples:
#   bash intermediate_layer_experiment/scripts/train_prompt_tuning_fixed.sh 3 20 1000 256
#   bash intermediate_layer_experiment/scripts/train_prompt_tuning_fixed.sh 3 20    # uses default 576 (no pruning)
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
LAYER_IDX=${1:-3}              # Default to layer 3
NUM_PROMPT_TOKENS=${2:-20}     # Default to 20 prompt tokens
MAX_SAMPLES=${3:-}             # Default to all samples 
VISUAL_TOKEN_NUM=${4:-576}     # Default to 576 

# Data paths
QUESTION_FILE="./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
ANNOTATION_FILE="./playground/data/eval/textvqa/TextVQA_0.5.1_val.json"
IMAGE_FOLDER="./playground/data/eval/textvqa/train_images"

# Output directory
OUTPUT_DIR="./intermediate_layer_experiment/checkpoints"

# Training hyperparameters
BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8  # Effective batch size = 2 * 8 = 16
NUM_EPOCHS=20
LEARNING_RATE=5e-4
WEIGHT_DECAY=0.01   # Small weight decay for regularization
WARMUP_RATIO=0.1
NUM_WORKERS=4
SEED=42
CONV_MODE="vicuna_v1"
PROMPT_INIT_METHOD="from_vocab"  # More stable than random initialization

# Build max_samples argument
MAX_SAMPLES_ARG=""
if [ -n "$MAX_SAMPLES" ]; then
    MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

echo "=========================================="
echo "Prompt Tuning Training Configuration (FIXED VERSION)"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Intermediate Layer: $LAYER_IDX"
echo "Number of Prompt Tokens: $NUM_PROMPT_TOKENS"
echo "Prompt Init Method: $PROMPT_INIT_METHOD"
echo "Visual Token Num: $VISUAL_TOKEN_NUM (576 = no pruning)"
echo "Max Samples: ${MAX_SAMPLES:-All}"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation Steps: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "FIX: Training extracts intermediate layer from question only,"
echo "     then appends answer embeddings. This ensures consistency"
echo "     with inference (which also only has question)."
echo "=========================================="
echo ""

# Run training
echo "Starting training..."
CUDA_VISIBLE_DEVICES=1 $PYTHON -u intermediate_layer_experiment/scripts/train_prompt_tuning_fixed.py \
    --model-path "$MODEL_PATH" \
    --intermediate-layer-idx $LAYER_IDX \
    --num-prompt-tokens $NUM_PROMPT_TOKENS \
    --prompt-init-method "$PROMPT_INIT_METHOD" \
    --visual-token-num $VISUAL_TOKEN_NUM \
    --question-file "$QUESTION_FILE" \
    --annotation-file "$ANNOTATION_FILE" \
    --image-folder "$IMAGE_FOLDER" \
    --conv-mode "$CONV_MODE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --num-epochs $NUM_EPOCHS \
    --learning-rate $LEARNING_RATE \
    --weight-decay $WEIGHT_DECAY \
    --warmup-ratio $WARMUP_RATIO \
    --num-workers $NUM_WORKERS \
    --seed $SEED \
    $MAX_SAMPLES_ARG

echo ""
echo "=========================================="
echo "Training complete!"
echo "Checkpoint saved to: $OUTPUT_DIR/prompt_tokens_layer${LAYER_IDX}_n${NUM_PROMPT_TOKENS}_fixed.pt"
echo "=========================================="


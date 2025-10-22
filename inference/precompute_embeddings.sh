#!/bin/bash

# Precompute MLP-aligned embeddings for CC dataset
# This script processes images through LLaVA 7B + MLP alignment
# and saves the aligned embeddings for later inference

# GPU selection (default: 0)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# ============================================================================
# Configuration - Modify these paths according to your setup
# ============================================================================

# Model paths
MODEL_7B="checkpoints/llava-v1.5-7b"
MLP_CHECKPOINT="checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt"

# Dataset paths
CC_ROOT="playground/data/CC_200"
JSON_PATH="inference/200_test.json"

# Output directory
OUTPUT_DIR="inference/embeddings"

# Vision tower configuration
VISION_TOWER="openai/clip-vit-large-patch14-336"
MM_VISION_SELECT_LAYER=-2
MM_VISION_SELECT_FEATURE="patch"

# Processing options
BATCH_SIZE=8
NUM_WORKERS=4
MAX_SAMPLES= # Leave empty to process all samples, or set a number for testing

# Precision
USE_BF16="--bf16"  # Comment out to use float32

# ============================================================================
# Run preprocessing
# ============================================================================

echo "=========================================="
echo "Precomputing MLP-aligned embeddings"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Model 7B: $MODEL_7B"
echo "MLP checkpoint: $MLP_CHECKPOINT"
echo "CC root: $CC_ROOT"
echo "JSON path: $JSON_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

# Build command
CMD="python inference/precompute_embeddings.py \
  --model_7b $MODEL_7B \
  --mlp_checkpoint $MLP_CHECKPOINT \
  --cc_root $CC_ROOT \
  --json_path $JSON_PATH \
  --output_dir $OUTPUT_DIR \
  --vision_tower $VISION_TOWER \
  --mm_vision_select_layer $MM_VISION_SELECT_LAYER \
  --mm_vision_select_feature $MM_VISION_SELECT_FEATURE \
  --batch_size $BATCH_SIZE \
  --num_workers $NUM_WORKERS"

# Add optional arguments
if [ -n "$MAX_SAMPLES" ]; then
  CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$USE_BF16" ]; then
  CMD="$CMD $USE_BF16"
fi

# Execute
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "Embeddings saved to: $OUTPUT_DIR"
echo "=========================================="


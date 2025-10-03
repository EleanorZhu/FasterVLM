#!/bin/bash

# Run inference using precomputed MLP-aligned embeddings with LLaVA 13B
# This script loads precomputed embeddings and generates text responses

# ============================================================================
# Configuration - Modify these paths according to your setup
# ============================================================================

# Model path
MODEL_13B="checkpoints/llava-v1.5-13b"

# Data paths
EMBEDDINGS_DIR="inference/embeddings"
JSON_PATH="inference/blip_laion_cc_sbu_558k.json"

# Output path
OUTPUT_FILE="inference/results/inference_results.json"

# Generation parameters
CONV_MODE="llava_v1"
MAX_NEW_TOKENS=512
TEMPERATURE=0.2
TOP_P=0.7

# Processing options
BATCH_SIZE=1  # Currently only supports 1
MAX_SAMPLES=10  # Leave empty to process all samples, or set a number for testing

# Precision
USE_BF16="--bf16"  # Comment out to use float32

# ============================================================================
# Run inference
# ============================================================================

echo "=========================================="
echo "Running inference with LLaVA 13B"
echo "=========================================="
echo "Model 13B: $MODEL_13B"
echo "Embeddings dir: $EMBEDDINGS_DIR"
echo "JSON path: $JSON_PATH"
echo "Output file: $OUTPUT_FILE"
echo "=========================================="

# Build command
CMD="python inference/run_inference.py \
  --model_13b $MODEL_13B \
  --embeddings_dir $EMBEDDINGS_DIR \
  --json_path $JSON_PATH \
  --output_file $OUTPUT_FILE \
  --conv_mode $CONV_MODE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --batch_size $BATCH_SIZE"

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
echo "Inference complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "=========================================="


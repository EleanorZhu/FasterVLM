#!/bin/bash
# Run baseline inference with full LLaVA 13B model on 200 test samples

set -e

cd "$(dirname "$0")/.."

# GPU selection (default: 0)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Configuration
MODEL_13B="checkpoints/llava-v1.5-13b"  # Local checkpoint
JSON_PATH="13B_inference/blip_laion_cc_200_test_samples.json"
IMAGES_DIR="playground/data/CC_200"
OUTPUT_FILE="13B_inference/results/llava13b_baseline_inference.json"

# Generation parameters (matching tested configuration)
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0  # Greedy decoding (0.0 = deterministic, >0 = sampling)
TOP_P=0.7        # Nucleus sampling (only used when temperature > 0)
NUM_BEAMS=1      # Beam search (1 = no beam search)
CONV_MODE="llava_v1"

echo "=========================================="
echo "LLaVA 13B Baseline Inference"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Model: $MODEL_13B"
echo "Test samples: $JSON_PATH"
echo "Images: $IMAGES_DIR"
echo "Output: $OUTPUT_FILE"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Num beams: $NUM_BEAMS"
echo "Conv mode: $CONV_MODE"
echo "=========================================="
echo ""

# Run inference
python3 13B_inference/run_llava13b_baseline.py \
  --model_13b "$MODEL_13B" \
  --json_path "$JSON_PATH" \
  --images_dir "$IMAGES_DIR" \
  --output_file "$OUTPUT_FILE" \
  --conv_mode "$CONV_MODE" \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --num_beams $NUM_BEAMS \
  --bf16

echo ""
echo "Baseline inference complete!"
echo "Results saved to: $OUTPUT_FILE"


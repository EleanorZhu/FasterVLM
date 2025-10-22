#!/bin/bash
# Merge MLP-aligned and baseline inference results for comparison

set -e

cd "$(dirname "$0")/.."

# Default paths
MLP_RESULTS="inference/results/inference_results.json"
BASELINE_RESULTS="13B_inference/results/llava13b_baseline_inference.json"
OUTPUT="inference/results/comparison_results.json"

echo "=========================================="
echo "Merging Results for Comparison"
echo "=========================================="
echo "MLP results: $MLP_RESULTS"
echo "Baseline results: $BASELINE_RESULTS"
echo "Output: $OUTPUT"
echo "=========================================="
echo ""

python3 inference/merge_results_for_comparison.py \
  --mlp_results "$MLP_RESULTS" \
  --baseline_results "$BASELINE_RESULTS" \
  --output "$OUTPUT"

echo ""
echo "=========================================="
echo "Merge complete!"
echo "You can now use the comparison file for evaluation:"
echo "  $OUTPUT"
echo "=========================================="


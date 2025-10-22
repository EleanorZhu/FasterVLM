#!/bin/bash
# Evaluate and compare MLP-aligned vs baseline inference results

set -e

cd "$(dirname "$0")/.."

COMPARISON_FILE="inference/results/comparison_results.json"

echo "=========================================="
echo "Evaluation and Comparison"
echo "=========================================="
echo "Using comparison file: $COMPARISON_FILE"
echo ""

# Check if comparison file exists
if [ ! -f "$COMPARISON_FILE" ]; then
    echo "Error: Comparison file not found!"
    echo "Please run: bash inference/merge_results_for_comparison.sh"
    exit 1
fi

# 1. Evaluate MLP-generated against ground truth
echo "=========================================="
echo "1. Evaluating MLP-generated vs Ground Truth"
echo "=========================================="
python3 inference/evaluate_cc_results.py \
  --results_file "$COMPARISON_FILE" \
  --reference_field ground_truth \
  --generated_field mlp_generated \
  --output_file inference/results/mlp_vs_groundtruth_metrics.json

echo ""
echo ""

# 2. Evaluate baseline-generated against ground truth
echo "=========================================="
echo "2. Evaluating Baseline-generated vs Ground Truth"
echo "=========================================="
python3 inference/evaluate_cc_results.py \
  --results_file "$COMPARISON_FILE" \
  --reference_field ground_truth \
  --generated_field baseline_generated \
  --output_file inference/results/baseline_vs_groundtruth_metrics.json

echo ""
echo ""

# 3. Compare MLP-generated against baseline-generated
echo "=========================================="
echo "3. Comparing MLP-generated vs Baseline-generated"
echo "=========================================="
echo "(Using baseline as reference)"
python3 inference/evaluate_cc_results.py \
  --results_file "$COMPARISON_FILE" \
  --reference_field baseline_generated \
  --generated_field mlp_generated \
  --output_file inference/results/mlp_vs_baseline_metrics.json

echo ""
echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo "Results saved to:"
echo "  1. inference/results/mlp_vs_groundtruth_metrics.json"
echo "  2. inference/results/baseline_vs_groundtruth_metrics.json"
echo "  3. inference/results/mlp_vs_baseline_metrics.json"
echo "=========================================="


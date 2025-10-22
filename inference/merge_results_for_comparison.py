#!/usr/bin/env python3
"""
Merge ground truth from MLP-aligned inference with generated text from baseline inference.
This creates a comparison file with both results side-by-side for evaluation.
"""

import json
import argparse
from pathlib import Path


def merge_results(mlp_results_path, baseline_results_path, output_path):
    """
    Merge ground truth from MLP results with generated text from baseline results.
    
    Args:
        mlp_results_path: Path to MLP-aligned inference results (contains ground_truth)
        baseline_results_path: Path to baseline inference results (contains generated)
        output_path: Path to save merged comparison results
    """
    # Load both result files
    print(f"Loading MLP-aligned results from: {mlp_results_path}")
    with open(mlp_results_path, 'r') as f:
        mlp_results = json.load(f)
    
    print(f"Loading baseline results from: {baseline_results_path}")
    with open(baseline_results_path, 'r') as f:
        baseline_results = json.load(f)
    
    # Create lookup dictionary for baseline results by image_id
    baseline_dict = {item['image_id']: item for item in baseline_results}
    
    # Merge results
    merged_results = []
    matched = 0
    missing = 0
    
    for mlp_item in mlp_results:
        image_id = mlp_item['image_id']
        
        if image_id in baseline_dict:
            baseline_item = baseline_dict[image_id]
            
            # Create merged entry
            merged_entry = {
                'image_id': image_id,
                'prompt': mlp_item.get('prompt', ''),
                'ground_truth': mlp_item.get('ground_truth', ''),
                'mlp_generated': mlp_item.get('generated', ''),
                'baseline_generated': baseline_item.get('generated', '')
            }
            merged_results.append(merged_entry)
            matched += 1
        else:
            print(f"Warning: Image {image_id} not found in baseline results")
            missing += 1
    
    # Save merged results
    print(f"\nMerging complete:")
    print(f"  Matched samples: {matched}")
    print(f"  Missing in baseline: {missing}")
    print(f"  Total merged: {len(merged_results)}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"\nMerged results saved to: {output_path}")
    
    # Print sample
    if merged_results:
        print("\nSample merged entry:")
        sample = merged_results[0]
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Prompt: {sample['prompt'][:80]}...")
        print(f"  Ground truth: {sample['ground_truth'][:80]}...")
        print(f"  MLP generated: {sample['mlp_generated'][:80]}...")
        print(f"  Baseline generated: {sample['baseline_generated'][:80]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Merge MLP-aligned and baseline inference results for comparison'
    )
    parser.add_argument(
        '--mlp_results',
        type=str,
        default='inference/results/inference_results.json',
        help='Path to MLP-aligned inference results (contains ground_truth)'
    )
    parser.add_argument(
        '--baseline_results',
        type=str,
        default='13B_inference/results/llava13b_baseline_inference.json',
        help='Path to baseline inference results (contains generated)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='inference/results/comparison_results.json',
        help='Path to save merged comparison results'
    )
    
    args = parser.parse_args()
    
    merge_results(args.mlp_results, args.baseline_results, args.output)


if __name__ == '__main__':
    main()


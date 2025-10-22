"""
Evaluation script for CC dataset VQA results.

This script evaluates generated captions against ground truth or compares two generated results.

Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- SPICE (if available)

Usage:
    # Evaluate against ground truth
    python inference/evaluate_cc_results.py \
        --results_file inference/results/inference_results.json \
        --reference_field ground_truth \
        --generated_field generated

    # Compare two generated results (e.g., MLP vs baseline)
    python inference/evaluate_cc_results.py \
        --results_file inference/results/comparison_results.json \
        --reference_field baseline_generated \
        --generated_field mlp_generated
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def load_results(results_file: str) -> List[Dict]:
    """Load inference results JSON file."""
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} samples")
    return results


def evaluate_with_pycocoevalcap(results: List[Dict], reference_field: str, generated_field: str) -> Dict[str, float]:
    """
    Evaluate using pycocoevalcap metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE).

    Args:
        results: List of result dictionaries
        reference_field: Field name for reference text (e.g., 'ground_truth', 'baseline_generated')
        generated_field: Field name for generated text to evaluate (e.g., 'generated', 'mlp_generated')

    Requires: pip install git+https://github.com/ronghanghu/coco-caption.git@python23
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ModuleNotFoundError:
        print("\n" + "="*60)
        print("ERROR: pycocoevalcap not found!")
        print("="*60)
        print("Please install it with:")
        print("  pip install git+https://github.com/ronghanghu/coco-caption.git@python23")
        print("\nOr try the coco-caption package:")
        print("  pip install pycocoevalcap")
        print("="*60 + "\n")
        raise

    # Try to import SPICE (optional, requires Java)
    try:
        from pycocoevalcap.spice.spice import Spice
        spice_available = True
    except (ModuleNotFoundError, ImportError):
        spice_available = False
        print("Warning: SPICE metric not available (requires Java 1.8.0)")

    print("\n" + "="*60)
    print("Preparing data for evaluation...")
    print(f"  Reference field: {reference_field}")
    print(f"  Generated field: {generated_field}")
    print("="*60)

    # Prepare data in COCO format
    # Format: {image_id: [{"caption": "..."}]}
    gts = {}  # reference (ground truth or baseline)
    res = {}  # results (generated to evaluate)

    skipped = 0
    for idx, entry in enumerate(results):
        image_id = entry['image_id']

        # Get reference and generated text
        ref = entry.get(reference_field)
        gen = entry.get(generated_field)

        # Skip if either field is missing
        if ref is None or gen is None:
            skipped += 1
            continue

        # Convert to list if needed
        if isinstance(ref, str):
            ref = [ref]

        gts[image_id] = [{"caption": caption} for caption in ref]
        res[image_id] = [{"caption": gen}]

    print(f"Prepared {len(gts)} samples for evaluation")
    if skipped > 0:
        print(f"Skipped {skipped} samples with missing fields")

    # Tokenize
    print("\nTokenizing captions...")
    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts)
    res_tokenized = tokenizer.tokenize(res)

    # Compute metrics
    metrics = {}

    print("\n" + "="*60)
    print("Computing metrics...")
    print("="*60)

    # BLEU
    print("\n[1/5] Computing BLEU scores...")
    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics['BLEU-1'] = bleu_scores[0]
    metrics['BLEU-2'] = bleu_scores[1]
    metrics['BLEU-3'] = bleu_scores[2]
    metrics['BLEU-4'] = bleu_scores[3]

    # METEOR
    print("[2/5] Computing METEOR score...")
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics['METEOR'] = meteor_score

    # ROUGE-L
    print("[3/5] Computing ROUGE-L score...")
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics['ROUGE-L'] = rouge_score

    # CIDEr
    print("[4/5] Computing CIDEr score...")
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    metrics['CIDEr'] = cider_score

    # SPICE (optional)
    if spice_available:
        print("[5/5] Computing SPICE score (requires Java)...")
        try:
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(gts_tokenized, res_tokenized)
            metrics['SPICE'] = spice_score
        except Exception as e:
            print(f"  Warning: SPICE computation failed: {e}")
            metrics['SPICE'] = None
    else:
        print("[5/5] SPICE: Skipped (not available)")
        metrics['SPICE'] = None

    return metrics


def evaluate_simple_metrics(results: List[Dict], reference_field: str, generated_field: str) -> Dict[str, float]:
    """
    Compute simple baseline metrics without external dependencies.

    Args:
        results: List of result dictionaries
        reference_field: Field name for reference text
        generated_field: Field name for generated text to evaluate
    """
    from collections import Counter

    print("\n" + "="*60)
    print("Computing simple baseline metrics...")
    print(f"  Reference field: {reference_field}")
    print(f"  Generated field: {generated_field}")
    print("="*60)

    metrics = {}

    # Average generation length
    gen_lengths = []
    ref_lengths = []

    for r in results:
        gen = r.get(generated_field)
        ref = r.get(reference_field)

        if gen:
            gen_lengths.append(len(gen.split()))
        if ref:
            if isinstance(ref, list):
                ref_lengths.extend([len(g.split()) for g in ref])
            else:
                ref_lengths.append(len(ref.split()))

    if gen_lengths:
        metrics['avg_gen_length'] = sum(gen_lengths) / len(gen_lengths)
    if ref_lengths:
        metrics['avg_ref_length'] = sum(ref_lengths) / len(ref_lengths)

    # Simple word overlap (unigram F1)
    precisions = []
    recalls = []
    for r in results:
        ref = r.get(reference_field)
        gen = r.get(generated_field)

        if not ref or not gen:
            continue

        if isinstance(ref, list):
            ref = ' '.join(ref)

        gen_words = set(gen.lower().split())
        ref_words = set(ref.lower().split())

        if len(gen_words) > 0:
            overlap = len(gen_words & ref_words)
            precision = overlap / len(gen_words)
            precisions.append(precision)

        if len(ref_words) > 0:
            overlap = len(gen_words & ref_words)
            recall = overlap / len(ref_words)
            recalls.append(recall)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0

    if avg_precision + avg_recall > 0:
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        f1 = 0

    metrics['word_precision'] = avg_precision
    metrics['word_recall'] = avg_recall
    metrics['word_f1'] = f1

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Pretty print metrics."""
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

    for metric_name, score in metrics.items():
        if score is not None:
            print(f"{metric_name:>20}: {score:.4f}")
        else:
            print(f"{metric_name:>20}: N/A")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CC dataset VQA results')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to inference results JSON file')
    parser.add_argument('--reference_field', type=str, default='ground_truth',
                        help='Field name for reference text (default: ground_truth). '
                             'Use "baseline_generated" to compare against baseline.')
    parser.add_argument('--generated_field', type=str, default='generated',
                        help='Field name for generated text to evaluate (default: generated). '
                             'Use "mlp_generated" to evaluate MLP results.')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation metrics JSON (optional)')
    parser.add_argument('--simple_only', action='store_true',
                        help='Only compute simple metrics (no pycocoevalcap required)')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_file)

    if len(results) == 0:
        print("Error: No results found in file!")
        return 1

    # Check if fields exist
    sample = results[0]
    if args.reference_field not in sample:
        print(f"\nError: Reference field '{args.reference_field}' not found in results!")
        print(f"Available fields: {list(sample.keys())}")
        return 1
    if args.generated_field not in sample:
        print(f"\nError: Generated field '{args.generated_field}' not found in results!")
        print(f"Available fields: {list(sample.keys())}")
        return 1

    print(f"\nEvaluating '{args.generated_field}' against '{args.reference_field}'")

    # Evaluate
    all_metrics = {}

    if not args.simple_only:
        try:
            # Try to use pycocoevalcap for standard metrics
            coco_metrics = evaluate_with_pycocoevalcap(results, args.reference_field, args.generated_field)
            all_metrics.update(coco_metrics)
            print_metrics(coco_metrics, "COCO Evaluation Metrics")
        except ModuleNotFoundError:
            print("\nFalling back to simple metrics (pycocoevalcap not available)")
            args.simple_only = True

    # Always compute simple metrics as well
    simple_metrics = evaluate_simple_metrics(results, args.reference_field, args.generated_field)
    all_metrics.update(simple_metrics)
    print_metrics(simple_metrics, "Simple Baseline Metrics")

    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        print(f"\n✓ Metrics saved to: {args.output_file}")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total samples evaluated: {len(results)}")

    if 'BLEU-4' in all_metrics:
        print(f"\nKey metrics:")
        print(f"  BLEU-4:  {all_metrics['BLEU-4']:.4f}")
        print(f"  METEOR:  {all_metrics['METEOR']:.4f}")
        print(f"  ROUGE-L: {all_metrics['ROUGE-L']:.4f}")
        print(f"  CIDEr:   {all_metrics['CIDEr']:.4f}")
        if all_metrics.get('SPICE') is not None:
            print(f"  SPICE:   {all_metrics['SPICE']:.4f}")

    print("="*60 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())

"""
Select 200 test samples from CC dataset for inference evaluation.

This script:
1. Loads training image IDs to ensure no overlap
2. Randomly selects 200 samples from the test-only annotations
3. Copies corresponding images to CC_200 folder
4. Creates a filtered JSON with only these 200 samples for inference

Ensures clean evaluation with no data leakage from MLP training.
"""

import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict


def load_training_ids(training_ids_path: str) -> set:
    """Load the set of training image IDs to exclude."""
    print(f"Loading training IDs from {training_ids_path}...")
    with open(training_ids_path, 'r') as f:
        training_ids = set(line.strip() for line in f if line.strip())
    print(f"  Loaded {len(training_ids)} training image IDs to exclude")
    return training_ids


def load_test_annotations(json_path: str) -> List[Dict]:
    """Load test-only annotations (already filtered to exclude training samples)."""
    print(f"\nLoading test annotations from {json_path}...")
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    print(f"  Total test samples available: {len(annotations)}")
    return annotations


def select_random_samples(
    annotations: List[Dict],
    num_samples: int,
    seed: int = 42
) -> List[Dict]:
    """Randomly select a subset of samples with fixed seed for reproducibility."""
    print(f"\nSelecting {num_samples} random samples (seed={seed})...")
    random.seed(seed)
    
    if num_samples > len(annotations):
        print(f"  Warning: Requested {num_samples} samples but only {len(annotations)} available")
        num_samples = len(annotations)
    
    selected = random.sample(annotations, num_samples)
    print(f"  Selected {len(selected)} samples")
    return selected


def verify_and_copy_images(
    selected_samples: List[Dict],
    cc_root: Path,
    output_image_dir: Path
) -> tuple[List[Dict], int]:
    """
    Verify image files exist and copy them to output directory.
    
    Returns:
        (valid_samples, missing_count): List of samples with valid images and count of missing
    """
    print(f"\nVerifying image files in {cc_root}...")
    print(f"Copying images to {output_image_dir}...")
    
    output_image_dir.mkdir(parents=True, exist_ok=True)
    
    valid_samples = []
    missing_count = 0
    
    for idx, sample in enumerate(selected_samples):
        # Image path is relative to CC root (e.g., "00453/004539375.jpg")
        rel_path = sample['image']
        source_path = cc_root / rel_path
        
        if source_path.exists():
            # Copy image to output directory with same filename
            image_id = sample['id']
            # Get file extension from original
            ext = source_path.suffix
            dest_path = output_image_dir / f"{image_id}{ext}"
            
            # Copy the image
            shutil.copy2(source_path, dest_path)
            
            # Update the image path in the sample to point to new location
            # Keep just the filename since all images are now in one flat directory
            sample_copy = sample.copy()
            sample_copy['image'] = f"{image_id}{ext}"
            valid_samples.append(sample_copy)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(selected_samples)} images...")
        else:
            missing_count += 1
            if missing_count <= 5:  # Print first few missing files
                print(f"  Warning: Image not found: {source_path}")
    
    print(f"  Valid images found and copied: {len(valid_samples)}")
    if missing_count > 0:
        print(f"  Missing images: {missing_count}")
    
    return valid_samples, missing_count


def save_filtered_json(
    samples: List[Dict],
    output_path: Path
):
    """Save the filtered samples to JSON file."""
    print(f"\nSaving filtered annotations to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"  Saved {len(samples)} samples")


def print_summary(
    total_test_samples: int,
    num_selected: int,
    num_valid: int,
    num_missing: int,
    output_json: Path,
    output_images: Path
):
    """Print final summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total test samples available:     {total_test_samples:,}")
    print(f"Samples randomly selected:        {num_selected}")
    print(f"Valid images found and copied:    {num_valid}")
    print(f"Missing images:                   {num_missing}")
    print(f"Success rate:                     {100 * num_valid / num_selected:.1f}%")
    print(f"\nOutput JSON:                      {output_json}")
    print(f"Output images directory:          {output_images}")
    print(f"  (Contains {num_valid} image files)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Select 200 test samples from CC dataset for inference evaluation"
    )
    parser.add_argument(
        "--training_ids",
        type=str,
        default="train_mlp/training_image_ids.txt",
        help="Text file with training image IDs to exclude",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        default="inference/blip_laion_cc_sbu_558k_test_only.json",
        help="JSON with test-only annotations (training samples already excluded)",
    )
    parser.add_argument(
        "--cc_root",
        type=str,
        default="playground/data/CC",
        help="Root directory of CC dataset",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="13B_inference/blip_laion_cc_200_test_samples.json",
        help="Output JSON file with 200 selected samples",
    )
    parser.add_argument(
        "--output_images",
        type=str,
        default="playground/data/CC_200",
        help="Output directory to copy the 200 test images",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection",
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    cc_root = Path(args.cc_root)
    output_json = Path(args.output_json)
    output_images = Path(args.output_images)
    
    # Step 1: Load training IDs (for verification/info only, test_json already excludes them)
    training_ids = load_training_ids(args.training_ids)
    
    # Step 2: Load test-only annotations
    test_annotations = load_test_annotations(args.test_json)
    
    # Step 3: Randomly select samples
    selected_samples = select_random_samples(
        test_annotations,
        args.num_samples,
        args.seed
    )
    
    # Step 4: Verify images exist and copy to output directory
    valid_samples, missing_count = verify_and_copy_images(
        selected_samples,
        cc_root,
        output_images
    )
    
    # Step 5: Save filtered JSON
    save_filtered_json(valid_samples, output_json)
    
    # Step 6: Print summary
    print_summary(
        total_test_samples=len(test_annotations),
        num_selected=len(selected_samples),
        num_valid=len(valid_samples),
        num_missing=missing_count,
        output_json=output_json,
        output_images=output_images
    )
    
    # Verify no overlap with training set (sanity check)
    selected_ids = {s['id'] for s in valid_samples}
    overlap = selected_ids & training_ids
    if overlap:
        print(f"\n⚠️  WARNING: Found {len(overlap)} samples overlapping with training set!")
        print(f"   This should not happen. First few overlapping IDs: {list(overlap)[:5]}")
    else:
        print(f"\n✅ Verified: No overlap with {len(training_ids)} training samples")


if __name__ == "__main__":
    main()


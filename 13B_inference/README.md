# 13B Inference Evaluation

This directory contains scripts and data for evaluating the MLP-aligned 7B→13B inference pipeline on a clean test set.

## Overview

The evaluation uses 200 test samples from the CC dataset that were **NOT** used during MLP training, ensuring no data leakage.

## Files

- `select_test_samples.py` - Script to select 200 random test samples and copy images
- `blip_laion_cc_200_test_samples.json` - Annotations for the 200 test samples
- `playground/data/CC_200/` - Directory containing the 200 test images

## Test Set Details

- **Total test samples available**: 458,128 (from CC dataset, excluding 100k training samples)
- **Selected for evaluation**: 200 samples (randomly selected with seed=42)
- **Success rate**: 100% (all 200 images found and copied)
- **Verified**: Zero overlap with 100,000 MLP training samples

## Data Structure

### JSON Format
Each entry in `blip_laion_cc_200_test_samples.json` contains:
```json
{
  "id": "001166603",
  "image": "001166603.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nRender a clear and concise summary of the photo."
    },
    {
      "from": "gpt",
      "value": "the logo for the new tv series"
    }
  ]
}
```

### Image Directory
- Location: `playground/data/CC_200/`
- Format: Images named by their ID (e.g., `001166603.jpg`)
- Total: 200 images

## Evaluation Pipelines

There are two inference pipelines to compare:

### Pipeline A: MLP-Aligned 7B→13B (FasterVLM)
- Image → 7B Vision Encoder → 7B Projector → **MLP Alignment** → 13B LLM
- Uses the trained MLP to align 7B visual features to 13B space

### Pipeline B: Full 13B Baseline
- Image → 13B Vision Encoder → 13B Projector → 13B LLM
- Standard LLaVA 13B inference (no alignment layer)

## Usage

### 1. Select Test Samples (Already Done)

```bash
python3 13B_inference/select_test_samples.py \
  --training_ids train_mlp/training_image_ids.txt \
  --test_json inference/blip_laion_cc_sbu_558k_test_only.json \
  --cc_root playground/data/CC \
  --output_json 13B_inference/blip_laion_cc_200_test_samples.json \
  --output_images playground/data/CC_200 \
  --num_samples 200 \
  --seed 42
```

### 2A. Run Baseline Inference (Full 13B Model)

Run the standard LLaVA 13B model as a baseline:

```bash
# Using shell script (recommended)
bash 13B_inference/run_llava13b_baseline.sh

# Or run directly
python3 13B_inference/run_llava13b_baseline.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --json_path 13B_inference/blip_laion_cc_200_test_samples.json \
  --images_dir playground/data/CC_200 \
  --output_file 13B_inference/results/llava13b_baseline_inference.json \
  --max_new_tokens 576 \
  --temperature 0.0 \
  --bf16
```

**Expected time**: ~20-30 minutes for 200 samples

### 2B. Run MLP-Aligned Inference (7B→13B Pipeline)

#### Step 1: Precompute MLP-Aligned Embeddings

```bash
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC_200 \
  --json_path 13B_inference/blip_laion_cc_200_test_samples.json \
  --output_dir 13B_inference/embeddings \
  --batch_size 8 \
  --bf16
```

**Expected time**: ~1-2 minutes for 200 samples

#### Step 2: Run 13B Inference with Aligned Embeddings

```bash
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir 13B_inference/embeddings \
  --json_path 13B_inference/blip_laion_cc_200_test_samples.json \
  --output_file 13B_inference/results/mlp_aligned_inference.json \
  --max_new_tokens 576 \
  --temperature 0.0 \
  --bf16
```

**Expected time**: ~20-30 minutes for 200 samples

### 3. Compare Results

After running both pipelines, you'll have:
- `13B_inference/results/llava13b_baseline_inference.json` - Full 13B baseline
- `13B_inference/results/mlp_aligned_inference.json` - MLP-aligned 7B→13B

Compare the generated captions to evaluate if the MLP alignment preserves quality.

## Expected Performance

### Baseline Inference (Full 13B)
- **Time**: ~20-30 minutes (200 samples)
- **GPU Memory**: ~25-30GB
- **Output**: JSON with generated captions from full 13B model

### MLP-Aligned Inference (7B→13B)

#### Precompute Embeddings
- **Time**: ~1-2 minutes (200 images)
- **GPU Memory**: ~10-15GB
- **Output**: 200 .pt files in `13B_inference/embeddings/`

#### Inference
- **Time**: ~20-30 minutes (200 samples)
- **GPU Memory**: ~25-30GB
- **Output**: JSON with generated captions from MLP-aligned pipeline

## Reproducibility

- Random seed: 42 (fixed for reproducible sample selection)
- No overlap with training data (verified)
- Same JSON format as full inference pipeline for compatibility

## Notes

- The 200 samples are a representative subset of the 458k test samples
- Images are copied to `CC_200/` for convenience and isolation
- JSON paths are updated to point to the copied images
- All samples verified to have valid image files before inclusion


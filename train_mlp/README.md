# MLP Alignment Pipeline (Contrastive Training)

This directory contains a two-step pipeline to train a lightweight MLP that aligns Stage‑1 (e.g., LLaVA‑v1.5‑7B) vision features to Stage‑2 (e.g., LLaVA‑v1.5‑13B) vision feature space using cosine similarity (contrastive) loss.

- Step 1: Precompute (X, y) pairs with frozen LLaVA encoders and projectors
- Step 2: Train the MLP on precomputed pairs with cosine similarity loss

Only the MLP is trained. LLaVA models are used only for offline feature extraction.

## Directory

- `generate_embeddings.py` — Create precomputed pairs `{x: (N, 4096), y: (N, 5120)}` from images
- `train_mlp.py` — Train the MLP alignment module on precomputed pairs
- `alignment.py` — Defines `MLPAlignment` and `cosine_alignment_loss`
- `train_mlp.sh` — Example shell script to launch training

## Requirements

- Python environment with PyTorch, Transformers, and PIL
- Two LLaVA model checkpoints/IDs:
  - `--model_7b` (e.g., llava-hf/llava-1.5-7b)
  - `--model_13b` (e.g., llava-hf/llava-1.5-13b)
- GPU recommended; bfloat16 optional

Note: Scripts assume execution from this folder (`train_mlp/`). They also insert the repo root into `sys.path` so `import llava` works from here.

## Dataset to use

- Use the CC dataset located in this repository at `playground/data/CC`.
- It contains many numbered subfolders (e.g., `00000`, `00001`, ...). `generate_embeddings.py` scans recursively and will find images under all subfolders.


## Step 1 — Precompute embeddings

Frozen feature extraction for both models on the same images, saving a pair per image.

Command:

```bash
cd train_mlp/
python generate_embeddings.py \
  --model_7b ../checkpoints/llava-v1.5-7b \
  --model_13b ../checkpoints/llava-v1.5-13b \
  --images_dir ../playground/data/CC \
  --output_dir ../playground/data/CC_pairs \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_vision_select_layer -2 \
  --mm_vision_select_feature patch \
  --bf16 true \
  --max_images 100000
```

Outputs:
- Pairs saved under `/path/to/output/embeddings/pairs/*.pt`
- Each file contains: `{ 'x': (N_tokens, 4096), 'y': (N_tokens, 5120), 'image_path': <str> }`

## Step 2 — Train the MLP

Train only the alignment MLP on precomputed tokens using cosine similarity loss.

Option A: Use the convenience script

```bash
cd train_mlp/
# Edit train_mlp.sh to set DATASET_PATH to the Step 1 pairs folder (e.g., ../playground/data/CC_pairs/pairs)
bash train_mlp.sh
```

Option B: Call the trainer directly

```bash
cd train_mlp/
python train_mlp.py \
  --dataset_path ../playground/data/CC_pairs/pairs \
  --output_dir ../checkpoints/contrastive-mlp-cc \
  --num_train_epochs 100 \
  --per_device_train_batch_size 256 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --report_to wandb \
  --run_name mlp-alignment-cc
```

Artifacts:
- Latest: `../checkpoints/contrastive-mlp-cc/mlp_alignment.pt`
- Periodic checkpoints under `../checkpoints/contrastive-mlp-cc/checkpoint-*/`

## Step 3 — Evaluate a trained MLP checkpoint

Use the evaluation script to compute cosine similarity, MSE, and contrastive loss on the validation/test split of your precomputed pairs. The script automatically reconstructs the MLP architecture from the checkpoint’s saved training args (e.g., residual adapter settings), so you don’t need to pass model flags again.

Required arguments:
- `--checkpoint_path`: Path to a saved checkpoint (e.g., best_mlp_alignment.pt)
- `--dataset_path`: Folder containing the precomputed pair `.pt` files with keys `{x, y}`

Optional arguments:
- `--batch_size` (default: 256)
- `--seed` (default: 42)
- `--dataloader_num_workers` (default: 0)
- `--verbose` (print per-batch stats)
- `--output_json` (path to write JSON summary; defaults beside the checkpoint)

Examples:

From the repository root:
```bash
python train_mlp/test_mlp.py \
  --checkpoint_path checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --dataset_path playground/data/CC_pairs/pairs \
  --batch_size 64
```

From inside the train_mlp/ directory:
```bash
cd train_mlp/
python test_mlp.py \
  --checkpoint_path ../checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --dataset_path ../playground/data/CC_pairs/pairs \
  --batch_size 64
```

Output:
- Console summary of mean/std for cosine similarity, MSE, and contrastive loss
- A JSON report saved next to the checkpoint (or to `--output_json` if provided)

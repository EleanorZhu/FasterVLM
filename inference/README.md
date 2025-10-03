# Inference Pipeline Command Line Cheatsheet

Quick reference for running the MLP-aligned embedding preprocessing and inference pipeline.

---

## 📋 Prerequisites

```bash
# Activate environment
conda activate fastervlm  # or your environment name

# Navigate to project root
cd /path/to/FasterVLM
```

---

## 🔧 Precompute Embeddings (Stage 1)

### Basic Usage

```bash
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_output \
  --batch_size 4 \
  --num_workers 4 \
  --bf16
```

### Test on Small Subset

```bash
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_test \
  --batch_size 2 \
  --num_workers 0 \
  --max_samples 10 \
  --bf16
```

### Full Dataset (558k images)

```bash
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_full \
  --batch_size 8 \
  --num_workers 8 \
  --bf16
```

### Using Shell Script

```bash
cd inference
bash precompute_embeddings.sh
```

### Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--model_7b` | Path to LLaVA 7B model | Required | `checkpoints/llava-v1.5-7b` |
| `--mlp_checkpoint` | Path to MLP checkpoint | Required | `checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt` |
| `--cc_root` | Root directory of CC images | Required | `playground/data/CC` |
| `--json_path` | Path to annotations JSON | Required | `inference/blip_laion_cc_sbu_558k.json` |
| `--output_dir` | Where to save embeddings | Required | `inference/embeddings_output` |
| `--batch_size` | Batch size for processing | 4 | 4-8 (depends on GPU memory) |
| `--num_workers` | DataLoader workers | 4 | 4-8 |
| `--max_samples` | Limit number of samples | None | Use for testing |
| `--bf16` | Use bfloat16 precision | Flag | Recommended for speed |

### Expected Performance

- **Speed**: ~3 images/second
- **Full dataset**: ~52 hours (558k images)
- **GPU Memory**: ~10-15GB (batch_size=4)

---

## 🚀 Run Inference (Stage 2)

### Basic Usage

```bash
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir inference/embeddings_output \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_file inference/results/output.json \
  --max_new_tokens 128 \
  --temperature 0.2 \
  --top_p 0.7 \
  --bf16
```

### Test on Small Subset

```bash
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir inference/embeddings_test \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_file inference/results/test_output.json \
  --max_samples 10 \
  --max_new_tokens 64 \
  --temperature 0.2 \
  --bf16
```

### Full Dataset

```bash
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir inference/embeddings_full \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_file inference/results/full_output.json \
  --max_new_tokens 128 \
  --temperature 0.2 \
  --top_p 0.7 \
  --bf16
```

### Using Shell Script

```bash
cd inference
bash run_inference.sh
```

### Key Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--model_13b` | Path to LLaVA 13B model | Required | `checkpoints/llava-v1.5-13b` |
| `--embeddings_dir` | Directory with precomputed embeddings | Required | `inference/embeddings_output` |
| `--json_path` | Path to annotations JSON | Required | `inference/blip_laion_cc_sbu_558k.json` |
| `--output_file` | Where to save results | Required | `inference/results/output.json` |
| `--max_samples` | Limit number of samples | None | Use for testing |
| `--max_new_tokens` | Max tokens to generate | 512 | 64-256 |
| `--temperature` | Sampling temperature | 0.2 | 0.0-1.0 (0=greedy) |
| `--top_p` | Nucleus sampling | 0.7 | 0.7-0.95 |
| `--conv_mode` | Conversation template | llava_v1 | llava_v1 |
| `--bf16` | Use bfloat16 precision | Flag | Recommended |

### Expected Performance

- **Speed**: ~7 seconds/sample
- **Full dataset**: ~27 days single GPU (558k samples)
- **GPU Memory**: ~25-30GB

---

## 🔄 Complete Pipeline (Both Stages)

### Quick Test (10 samples)

```bash
# Stage 1: Precompute embeddings
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_test \
  --batch_size 2 \
  --max_samples 10 \
  --bf16

# Stage 2: Run inference
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir inference/embeddings_test \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_file inference/results/test_output.json \
  --max_samples 10 \
  --max_new_tokens 128 \
  --temperature 0.2 \
  --bf16
```

### Full Pipeline

```bash
# Stage 1: Precompute all embeddings (~52 hours)
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_full \
  --batch_size 8 \
  --num_workers 8 \
  --bf16

# Stage 2: Run inference on all samples (~27 days)
python3 inference/run_inference.py \
  --model_13b checkpoints/llava-v1.5-13b \
  --embeddings_dir inference/embeddings_full \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_file inference/results/full_output.json \
  --max_new_tokens 128 \
  --temperature 0.2 \
  --top_p 0.7 \
  --bf16
```

---

## 📊 Check Results

### View JSON Output

```bash
# Pretty print JSON
cat inference/results/output.json | python3 -m json.tool | less

# Count samples
cat inference/results/output.json | python3 -c "import json, sys; print(len(json.load(sys.stdin)))"

# View first 3 samples
python3 -c "import json; results = json.load(open('inference/results/output.json')); [print(f'Sample {i+1}:\n  Prompt: {r[\"prompt\"]}\n  Generated: {r[\"generated\"]}\n') for i, r in enumerate(results[:3])]"
```

### Check Embedding Files

```bash
# Count embedding files
ls inference/embeddings_output/*.pt | wc -l

# Check a single embedding
python3 -c "import torch; data = torch.load('inference/embeddings_output/004539375.pt'); print(f'Shape: {data[\"embedding\"].shape}, ID: {data[\"image_id\"]}')"
```

---

## 🛠️ Troubleshooting

### Out of Memory

```bash
# Reduce batch size for preprocessing
--batch_size 2

# Use smaller max_new_tokens for inference
--max_new_tokens 64
```

### Slow Performance

```bash
# Increase batch size (if memory allows)
--batch_size 8

# Increase workers for preprocessing
--num_workers 8

# Use bfloat16
--bf16
```

### Resume Interrupted Run

The scripts automatically skip existing embedding files, so you can safely re-run:

```bash
# Will skip already processed images
python3 inference/precompute_embeddings.py \
  --model_7b checkpoints/llava-v1.5-7b \
  --mlp_checkpoint checkpoints/contrastive-mlp-cc/best_mlp_alignment.pt \
  --cc_root playground/data/CC \
  --json_path inference/blip_laion_cc_sbu_558k.json \
  --output_dir inference/embeddings_output \
  --batch_size 4 \
  --bf16
```

---

## 📈 Performance Tips

### Multi-GPU Preprocessing

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python3 inference/precompute_embeddings.py \
  --output_dir inference/embeddings_gpu0 \
  --max_samples 279000 \
  ... other args ...

# GPU 1
CUDA_VISIBLE_DEVICES=1 python3 inference/precompute_embeddings.py \
  --output_dir inference/embeddings_gpu1 \
  --max_samples 279000 \
  ... other args ...

# Merge results
mkdir inference/embeddings_full
cp inference/embeddings_gpu0/*.pt inference/embeddings_full/
cp inference/embeddings_gpu1/*.pt inference/embeddings_full/
```

### Batch Inference Processing

Split the dataset and run multiple inference jobs:

```bash
# Process samples 0-100k
python3 inference/run_inference.py \
  --max_samples 100000 \
  --output_file inference/results/output_part1.json \
  ... other args ...

# Process samples 100k-200k (modify dataset to skip first 100k)
# Then merge JSON files
```

---

## 📝 Example Output Format

```json
[
  {
    "image_id": "004539375",
    "prompt": "Render a clear and concise summary of the photo.",
    "ground_truth": "select luxury furniture 3 - inch gel memory foam mattress topper",
    "generated": "The image features a large, well-made bed with a white comforter..."
  }
]
```

---

## ⚡ Quick Reference

```bash
# Test pipeline (10 samples, ~2 minutes total)
python3 inference/precompute_embeddings.py --max_samples 10 --output_dir inference/test_emb --bf16
python3 inference/run_inference.py --embeddings_dir inference/test_emb --max_samples 10 --output_file inference/results/test.json --bf16

# Full pipeline (558k samples)
python3 inference/precompute_embeddings.py --output_dir inference/embeddings_full --bf16
python3 inference/run_inference.py --embeddings_dir inference/embeddings_full --output_file inference/results/full.json --bf16
```


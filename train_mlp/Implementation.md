# Module-Based MLP Alignment Training

## 🎯 Overview
This implementation provides a two-step, module-based approach for training an MLP alignment layer that bridges LLaVA-v1.5 7B and 13B vision embeddings. By precomputing embeddings, we decouple data preparation from model training, enabling more efficient experimentation and reduced memory requirements during training.

## 🔄 Two-Step Pipeline

### Step 1: Dataset Construction
Precompute and save embeddings from both LLaVA models:
```
Images → LLaVA-v1.5-7B → X (4096-dim embeddings) → Save to disk
Images → LLaVA-v1.5-13B → y (5120-dim embeddings) → Save to disk
```

### Step 2: MLP Training
Load precomputed dataset and train alignment module:
```
Load (X, y) pairs → MLP Projector → Cosine Similarity Loss → Optimized MLP
```

## 📊 Dataset Structure

| Component | Source Model | Dimension | Role | Storage Format |
|-----------|-------------|-----------|------|----------------|
| **X** (Input) | LLaVA-v1.5-7B Vision Output | 4096 | MLP Input Features | `.pt` |
| **y** (Target) | LLaVA-v1.5-13B Vision Output | 5120 | Alignment Target | `.pt` |

### Dataset Preparation Details
- **Input Processing**: Images → Vision Encoder → Projector → Embeddings
- **Batch Processing**: Process images in batches to optimize memory usage
- **Storage**: Save as memory-mapped arrays for efficient loading
- **Preprocessing**: Both models use same image preprocessing pipeline

## 🏗️ MLP Alignment Module

### Architecture
```python
MLP Projector:
  Linear(4096 → 5120)
  LayerNorm(5120)
  GELU()
  Linear(5120 → 5120)
```

### Training Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| **Objective** | Cosine Similarity | Maximize similarity between aligned X and target y |
| **Loss Function** | `1 - cos_sim(MLP(X), y)` | Minimize distance in embedding space |
| **Optimizer** | AdamW | With weight decay for regularization |
| **Learning Rate** | 1e-4 | With cosine annealing schedule |
| **Batch Size** | 256 | Adjust based on available memory |
| **Training Mode** | MLP only | All other components remain frozen |

## 📈 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Cosine Similarity** | >0.9 | Average similarity between MLP(X) and y |
| **MSE** | <0.01 | Mean squared error in embedding space |
| **Alignment Variance** | <0.05 | Consistency across different image types |

### Validation Strategy
- Hold-out validation set (10-20% of data)
- Evaluate on diverse image categories
- Monitor overfitting through train/val loss curves


## 📝 Usage Workflow

```bash
# Step 1: Generate dataset
python generate_embeddings.py \
  --model_7b llava-hf/llava-1.5-7b-hf \
  --model_13b llava-hf/llava-1.5-13b-hf \
  --images_dir /path/to/images \
  --output_dir /path/to/embeddings

# Step 2: Train MLP
python train_mlp.py \
  --dataset_path /path/to/embeddings \
  --output_dir /path/to/mlp_checkpoint \
  --epochs 100 \
  --batch_size 256
```
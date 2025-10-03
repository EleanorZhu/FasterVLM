#!/bin/bash
cd "$(dirname "$0")"


# Train MLP alignment on precomputed (X, y) pairs
set -e

# Activate env (optional)
# eval "$(micromamba shell hook --shell bash)" && micromamba activate fastervlm

# Paths
DATASET_PATH="../playground/data/CC_pairs/pairs"   # folder of .pt with {x, y}
OUTPUT_DIR="../checkpoints/contrastive-mlp-cc"

# Hyperparameters per README
EPOCHS=100
BATCH_SIZE=64
LR=5e-5
WARMUP_RATIO=0.02
WEIGHT_DECAY=0.01

# Residual architecture (recommended to meet >0.99 cosine and <0.1 MSE)
MLP_NUM_BLOCKS=2          # 0 keeps original 2-layer MLP; 
MLP_BOTTLENECK=2048       # 5120 -> 1024 -> 5120 bottleneck inside adapters
MLP_ACTIVATION="gelu"     # relu|gelu
MLP_FINAL_LN=false         # apply final LayerNorm after adapters
MLP_OUT_ZERO_INIT=false    # zero-init final projection for stability
MLP_DROPOUT=0.0           # dropout used inside adapters (and legacy path)

# Logging
export WANDB_PROJECT="fastervlm-contrastive"
RUN_NAME="mlp-alignment-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUTPUT_DIR"

python train_mlp.py \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --learning_rate $LR \
  --warmup_ratio $WARMUP_RATIO \
  --weight_decay $WEIGHT_DECAY \
  --mlp_dropout $MLP_DROPOUT \
  --mlp_num_blocks $MLP_NUM_BLOCKS \
  --mlp_bottleneck_dim $MLP_BOTTLENECK \
  --mlp_residual_scale_init 1e-3 \
  --mlp_activation $MLP_ACTIVATION \
  --mlp_final_layernorm $MLP_FINAL_LN \
  --mlp_out_zero_init $MLP_OUT_ZERO_INIT \
  --dataloader_num_workers 0 \
  --report_to "wandb" \
  --run_name "$RUN_NAME"


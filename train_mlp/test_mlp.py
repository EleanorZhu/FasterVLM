import os
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Ensure repo root and this folder are on sys.path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))

from alignment import MLPAlignment  # noqa: E402
from train_mlp import PrecomputedPairs  # noqa: E402


def evaluate_batch(mlp: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for a batch:
      - cosine similarity (mean across tokens)
      - mse loss
      - contrastive loss = 1 - cosine similarity
    """
    with torch.no_grad():
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        pred = mlp(x_flat)

        pred_n = F.normalize(pred, dim=-1)
        y_n = F.normalize(y_flat, dim=-1)
        cos = (pred_n * y_n).sum(dim=-1).mean().float().item()
        mse = F.mse_loss(pred.float(), y_flat.float()).item()
        contrastive = 1.0 - cos

    return {"cosine_similarity": cos, "mse": mse, "contrastive": contrastive}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLP alignment checkpoint on precomputed pairs")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Folder containing .pt files with keys {x, y}")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="DataLoader workers (0 avoids shm issues)")
    parser.add_argument("--verbose", action="store_true", help="Print per-batch stats")
    parser.add_argument("--output_json", type=str, default=None, help="Where to write JSON results (default: beside checkpoint)")

    args = parser.parse_args()

    # Load checkpoint
    ckpt_obj = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = ckpt_obj.get("state_dict", ckpt_obj)
    ckpt_args = ckpt_obj.get("args", {}) if isinstance(ckpt_obj, dict) else {}

    # Derive model/config from checkpoint args if present
    mlp_input_dim = ckpt_args.get("mlp_input_dim", 4096)
    mlp_hidden_dim = ckpt_args.get("mlp_hidden_dim", 5120)
    mlp_output_dim = ckpt_args.get("mlp_output_dim", 5120)
    mlp_dropout = ckpt_args.get("mlp_dropout", 0.0)
    use_bf16 = bool(ckpt_args.get("bf16", False))

    # Split settings to complement training split
    val_ratio = float(ckpt_args.get("val_ratio", 0.2))
    seed = int(args.seed if args.seed is not None else ckpt_args.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Dataset & split: use the same deterministic split scheme as training and take the val part as test
    dataset = PrecomputedPairs(args.dataset_path, max_files=None)
    if val_ratio <= 0.0 or len(dataset) <= 1:
        # If no val split was used during training, evaluate on the full dataset
        test_dataset = dataset
        train_size = len(dataset)
        val_size = 0
    else:
        val_size = max(1, int(round(val_ratio * len(dataset))))
        train_size = max(1, len(dataset) - val_size)
        g = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(dataset, [train_size, val_size], generator=g)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    mlp = MLPAlignment(
        input_dim=mlp_input_dim,
        hidden_dim=mlp_hidden_dim,
        output_dim=mlp_output_dim,
        dropout=mlp_dropout,
    ).to(device=device, dtype=dtype)
    mlp.load_state_dict(state_dict)
    mlp.eval()

    # Evaluation loop
    batch_stats = []
    n_batches = 0
    for bx, by in test_loader:
        bx = bx.to(device=device, dtype=dtype)
        by = by.to(device=device, dtype=dtype)
        stats = evaluate_batch(mlp, bx, by)
        batch_stats.append(stats)
        n_batches += 1
        if args.verbose:
            print(f"batch {n_batches}: cos={stats['cosine_similarity']:.4f} mse={stats['mse']:.6f} contrastive={stats['contrastive']:.4f}")

    # Aggregate
    def mean_std(key: str):
        vals = torch.tensor([s[key] for s in batch_stats], dtype=torch.float64)
        return vals.mean().item() if len(vals) > 0 else float("nan"), vals.std(unbiased=False).item() if len(vals) > 0 else float("nan")

    mean_cos, std_cos = mean_std("cosine_similarity")
    mean_mse, std_mse = mean_std("mse")
    mean_contrastive, std_contrastive = mean_std("contrastive")

    total_pairs = len(test_dataset)

    results: Dict[str, Any] = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "num_test_files": int(total_pairs),
        "train_size": int(train_size),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
        "metrics": {
            "cosine_similarity": {"mean": mean_cos, "std": std_cos},
            "mse": {"mean": mean_mse, "std": std_mse},
            "contrastive": {"mean": mean_contrastive, "std": std_contrastive},
        },
    }

    print("\nEvaluation summary:")
    print(json.dumps(results["metrics"], indent=2))

    # Save JSON
    if args.output_json is None:
        ckpt_dir = Path(args.checkpoint_path).resolve().parent
        base = Path(args.checkpoint_path).stem
        out_path = ckpt_dir / f"eval_{base}.json"
    else:
        out_path = Path(args.output_json)
        if out_path.is_dir():
            base = Path(args.checkpoint_path).stem
            out_path = out_path / f"eval_{base}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved evaluation results to: {out_path}")


if __name__ == "__main__":
    main()


import os
import math
import argparse
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from alignment import MLPAlignment, combined_alignment_loss
import sys
from pathlib import Path

# Ensure repo root and this folder are on sys.path
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))



class PrecomputedPairs(Dataset):
    """
    Dataset of precomputed pairs saved as individual .pt files, each containing:
        {'x': (N_tokens, Din), 'y': (N_tokens, Dout)}
    where x are 7B-space features, y are 13B-space features.
    """
    def __init__(self, folder: str, max_files: int | None = None):
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.pt')]
        self.files.sort()
        if max_files is not None:
            self.files = self.files[:max_files]
        if not self.files:
            raise ValueError(f"No .pt pairs found in {folder}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        d = torch.load(self.files[idx], map_location='cpu')
        return d['x'], d['y']


def main():
    parser = argparse.ArgumentParser(description='Train MLP alignment on precomputed (X, y) pairs using cosine similarity loss.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Folder containing .pt files with keys {x, y}')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--per_device_train_batch_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--bf16', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--max_files', type=int, default=None, help='Limit number of pair files loaded')
    parser.add_argument('--report_to', type=str, default='none')
    parser.add_argument('--run_name', type=str, default='mlp_alignment')

    # MLP dims (per README: 4096 -> 5120 -> 5120)
    parser.add_argument('--mlp_input_dim', type=int, default=4096)
    parser.add_argument('--mlp_hidden_dim', type=int, default=5120)
    parser.add_argument('--mlp_output_dim', type=int, default=5120)
    parser.add_argument('--mlp_dropout', type=float, default=0.0)

    # Residual adapter architecture options (backward compatible defaults)
    parser.add_argument('--mlp_num_blocks', type=int, default=0, help='Number of residual adapter blocks (0 keeps original 2-layer MLP)')
    parser.add_argument('--mlp_bottleneck_dim', type=int, default=1024, help='Bottleneck dim inside residual adapters')
    parser.add_argument('--mlp_residual_scale_init', type=float, default=1e-3, help='Initial residual scale (LayerScale)')
    parser.add_argument('--mlp_activation', type=str, default='relu', help='Activation in residual adapters: relu|gelu')
    parser.add_argument('--mlp_final_layernorm', type=lambda x: str(x).lower() == 'true', default=False, help='Apply final LayerNorm after adapters')
    parser.add_argument('--mlp_out_zero_init', type=lambda x: str(x).lower() == 'true', default=True, help='Zero-init final projection when using adapters')
    parser.add_argument('--mlp_normalize_output', type=lambda x: str(x).lower() == 'true', default=False, help='L2-normalize outputs at the end')



    # Validation and early stopping options
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio (0-1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for dataset split')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (epochs)')
    parser.add_argument('--min_delta', type=float, default=0.0, help='Minimum improvement in val loss to reset patience')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    dataset = PrecomputedPairs(args.dataset_path, max_files=args.max_files)

    # Deterministic train/val split
    val_ratio = max(0.0, min(1.0, args.val_ratio))
    if val_ratio > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(val_ratio * len(dataset))))
        train_size = max(1, len(dataset) - val_size)
        g = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=g)
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            drop_last=False,
        )

    mlp = MLPAlignment(
        input_dim=args.mlp_input_dim,
        hidden_dim=args.mlp_hidden_dim,
        output_dim=args.mlp_output_dim,
        dropout=args.mlp_dropout,
        num_blocks=args.mlp_num_blocks,
        bottleneck_dim=args.mlp_bottleneck_dim,
        residual_scale_init=args.mlp_residual_scale_init,
        activation=args.mlp_activation,
        out_zero_init=args.mlp_out_zero_init,
        final_layernorm=args.mlp_final_layernorm,
        normalize_output=args.mlp_normalize_output,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = args.num_train_epochs * math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps))
    warmup_steps = int(args.warmup_ratio * total_steps)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_wandb = args.report_to.lower() == 'wandb'
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get('WANDB_PROJECT', 'fastervlm-contrastive'), name=args.run_name)
        wandb.config.update(vars(args))

    epochs_no_improve = 0

    global_step = 0
    mlp.train()

    stop_training = False
    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(args.num_train_epochs):
        mlp.train()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=dtype)

            x_flat = x.reshape(-1, x.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])

            pred = mlp(x_flat)
            loss = combined_alignment_loss(pred, y_flat)
            (loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    with torch.no_grad():
                        pred_n = torch.nn.functional.normalize(pred, dim=-1)
                        y_n = torch.nn.functional.normalize(y_flat, dim=-1)
                        cos = (pred_n * y_n).sum(dim=-1).mean().float().item()
                        mse = torch.nn.functional.mse_loss(pred.float(), y_flat.float()).item()
                    if use_wandb:
                        import wandb
                        wandb.log({'train/loss': loss.detach().float().item(),
                                   'train/cosine_similarity': cos,
                                   'train/mse': mse,
                                   'step': global_step})
                    print(f"epoch {epoch} step {global_step}: loss={loss.detach().float().item():.4f} cos={cos:.4f} mse={mse:.6f}")

                if global_step % args.save_steps == 0:
                    save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({'state_dict': mlp.state_dict(), 'args': vars(args)}, os.path.join(save_dir, 'mlp_alignment.pt'))

        # Epoch-end: save latest
        torch.save({'state_dict': mlp.state_dict(), 'args': vars(args)}, os.path.join(args.output_dir, 'mlp_alignment.pt'))

        # Validation
        if val_loader is not None:
            mlp.eval()
            val_loss_sum = 0.0
            val_cos_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device=device, dtype=dtype)
                    vy = vy.to(device=device, dtype=dtype)
                    vx_flat = vx.reshape(-1, vx.shape[-1])
                    vy_flat = vy.reshape(-1, vy.shape[-1])
                    vpred = mlp(vx_flat)
                    vloss = combined_alignment_loss(vpred, vy_flat)
                    val_loss_sum += vloss.detach().float().item()
                    vpred_n = torch.nn.functional.normalize(vpred, dim=-1)
                    vy_n = torch.nn.functional.normalize(vy_flat, dim=-1)
                    vcos = (vpred_n * vy_n).sum(dim=-1).mean().float().item()
                    val_cos_sum += vcos
                    val_batches += 1
            mean_val_loss = val_loss_sum / max(1, val_batches)
            mean_val_cos = val_cos_sum / max(1, val_batches)
            if use_wandb:
                import wandb
                wandb.log({'val/loss': mean_val_loss,
                           'val/cosine_similarity': mean_val_cos,
                           'epoch': epoch,
                           'step': global_step})
            print(f"epoch {epoch} validation: loss={mean_val_loss:.4f} cos={mean_val_cos:.4f}")

            # Early stopping
            improved = (best_val_loss - mean_val_loss) >= args.min_delta
            if improved or best_state_dict is None:
                best_val_loss = mean_val_loss
                # store CPU copy
                best_state_dict = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
                torch.save({'state_dict': mlp.state_dict(), 'args': vars(args)}, os.path.join(args.output_dir, 'best_mlp_alignment.pt'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                    stop_training = True
        if stop_training:
            break

    # Restore best model weights and save as latest
    if best_state_dict is not None:
        mlp.load_state_dict(best_state_dict)
        torch.save({'state_dict': mlp.state_dict(), 'args': vars(args)}, os.path.join(args.output_dir, 'mlp_alignment.pt'))

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()


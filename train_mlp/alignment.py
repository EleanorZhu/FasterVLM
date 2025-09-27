import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualAdapter(nn.Module):
    """
    Pre-LN residual adapter block with bottleneck: dim -> bottleneck -> dim
    Uses small residual scaling (LayerScale-style) for stability.
    """
    def __init__(
        self,
        dim: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
        activation: str = 'relu',
        residual_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        act = activation.lower()
        self.act = nn.ReLU() if act == 'relu' else nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        # Per-dimension residual scaling for stability
        self.gamma = nn.Parameter(torch.ones(dim) * residual_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.down(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        return x + self.gamma * h


class MLPAlignment(nn.Module):
    """
    Alignment module mapping Stage-1 (e.g., 7B) vision features to
    Stage-2 (e.g., 13B) vision feature space.

    Backward compatible with original 2-layer MLP when num_blocks == 0.
    When num_blocks > 0, applies residual adapter blocks between input and output projections.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        # New optional parameters (backward compatible)
        num_blocks: int = 0,
        bottleneck_dim: int = 1024,
        residual_scale_init: float = 1e-3,
        activation: str = 'relu',
        out_zero_init: bool = True,
        final_layernorm: bool = False,
        normalize_output: bool = False,
    ) -> None:
        super().__init__()
        assert hidden_dim == output_dim, (
            "Hidden dim should equal output dim per spec (5120). Got hidden_dim="
            f"{hidden_dim}, output_dim={output_dim}"
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.use_residual = num_blocks > 0

        self.in_proj = nn.Linear(input_dim, output_dim)

        if self.use_residual:
            # Residual adapter stack
            self.blocks = nn.ModuleList([
                ResidualAdapter(
                    dim=output_dim,
                    bottleneck_dim=bottleneck_dim,
                    dropout=dropout,
                    activation=activation,
                    residual_scale_init=residual_scale_init,
                )
                for _ in range(num_blocks)
            ])
            self.final_ln = nn.LayerNorm(output_dim) if final_layernorm else nn.Identity()
        else:
            # Original two-layer path: LN -> GELU -> Dropout
            self.norm = nn.LayerNorm(output_dim)
            self.act = nn.GELU()

        self.out_proj = nn.Linear(output_dim, output_dim)
        if self.use_residual and out_zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

        self.normalize_output = normalize_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., input_dim) 2D or 3D (e.g., [B, N, C_in]) is supported.
        Returns: (..., output_dim)
        """
        y = self.in_proj(x)

        if self.use_residual:
            for blk in self.blocks:
                y = blk(y)
            y = self.final_ln(y)
            y = self.out_proj(y)
        else:
            y = self.norm(y)
            y = self.act(y)
            y = self.dropout(y)
            y = self.out_proj(y)

        if self.normalize_output:
            y = F.normalize(y, p=2, dim=-1)
        return y


def cosine_alignment_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss: 1 - cos_sim(pred, target)
    pred: (..., D)
    target: (..., D)
    Returns scalar mean loss.
    """
    pred_flat = pred.reshape(-1, pred.shape[-1])
    tgt_flat = target.reshape(-1, target.shape[-1])
    pred_norm = F.normalize(pred_flat, p=2, dim=-1)
    tgt_norm = F.normalize(tgt_flat, p=2, dim=-1)
    cos = (pred_norm * tgt_norm).sum(dim=-1)
    return (1.0 - cos).mean()


def combined_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_cos: float = 1.0,
    mu_mse: float = 0.25,
) -> torch.Tensor:
    """
    Combined loss for alignment: L = lambda_cos * cosine_loss + mu_mse * MSE
    Defaults: lambda_cos=1.0, mu_mse=0.25
    """
    cos_loss = cosine_alignment_loss(pred, target)
    mse_loss = F.mse_loss(pred.float(), target.float())
    return lambda_cos * cos_loss + mu_mse * mse_loss


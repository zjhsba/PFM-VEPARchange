"""
Temporal Event Encoder for Event Camera Frame Aggregation

Replaces the simple .mean(dim=1) frame averaging with learnable temporal attention.
Implements factorized spatio-temporal processing:
  1. Per-patch temporal self-attention across frames (shared across spatial positions)
  2. Learnable frame weighting for aggregation
  3. Residual connection from original mean features

Architecture:
  Input:  [B, F, N, D]  (batch, frames, patches, dim)
  Output: [B, N, D]     (batch, patches, dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding1D(nn.Module):
    """1D sinusoidal positional encoding for temporal sequence positions."""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] input features
        Returns:
            position encoding [1, seq_len, D]
        """
        seq_len = x.shape[1]
        return self.pe[:, :seq_len, :]


class TemporalTransformerEncoder(nn.Module):
    """
    Lightweight temporal transformer encoder.
    Uses pre-norm architecture for better training stability.
    """

    def __init__(self, dim: int = 768, num_heads: int = 8,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 num_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.pos_encoding = PositionalEncoding1D(dim, max_len=32)
        self.pos_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TemporalTransformerLayer(dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B*N, F, D] factorized input (each spatial position is a batch item)
        Returns:
            [B*N, F, D] temporally enhanced features
        """
        x = x + self.pos_encoding(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


class TemporalTransformerLayer(nn.Module):
    """Single temporal transformer layer with pre-norm architecture."""

    def __init__(self, dim: int, num_heads: int, dim_feedforward: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + self.dropout1(attn_out)

        # MLP with pre-norm
        normed = self.norm2(x)
        mlp_out = self.mlp(normed)
        x = x + self.dropout2(mlp_out)

        return x


class TemporalEventEncoder(nn.Module):
    """
    Temporal encoder for event frame feature aggregation.

    Replaces simple frame averaging with:
    1. Factorized temporal self-attention (per spatial patch, across frames)
    2. Learnable per-frame importance weights for aggregation
    3. Optional residual connection from original mean features

    Args:
        dim: Feature dimension (default 768 for ViT-B)
        num_layers: Number of temporal transformer layers
        num_heads: Number of attention heads
        dim_feedforward: Feedforward dimension in transformer
        dropout: Dropout rate
        max_frames: Maximum number of event frames to support
        use_residual: Whether to add residual from original mean pooling
    """

    def __init__(
        self,
        dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 32,
        use_residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_residual = use_residual

        # Core temporal transformer
        self.temporal_transformer = TemporalTransformerEncoder(
            dim=dim,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_layers,
        )

        # Learnable frame weights for aggregation
        # Each frame gets a learnable importance score
        self.frame_weights = nn.Parameter(torch.zeros(max_frames, 1))
        nn.init.normal_(self.frame_weights, mean=0.0, std=0.02)

        # Post-aggregation refinement
        self.aggregate_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Event features of shape [B, F, N, D]
               B = batch size, F = number of frames,
               N = number of spatial patches, D = feature dimension

        Returns:
            Aggregated features of shape [B, N, D]
        """
        B, n_frames, N, D = x.shape

        # Save original mean-pooled features for residual
        residual = x.mean(dim=1) if self.use_residual else 0.0  # [B, N, D]

        # ---- Factorized temporal attention ----
        # Reshape: each spatial position is treated as an independent batch item
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, n_frames, D)  # [B*N, n_frames, D]

        # Temporal self-attention across frames
        x_flat = self.temporal_transformer(x_flat)  # [B*N, n_frames, D]

        # ---- Learnable frame weighting ----
        # Get frame weights for actual n_frames frames
        w = self.frame_weights[:n_frames]  # [n_frames, 1]
        w = F.softmax(w, dim=0)     # normalize across frames
        w = w.unsqueeze(0)          # [1, n_frames, 1]

        # Weighted sum over frames
        x_weighted = (x_flat * w).sum(dim=1)  # [B*N, D]

        # ---- Reshape back ----
        x_agg = x_weighted.reshape(B, N, D)  # [B, N, D]
        x_agg = self.aggregate_norm(x_agg)

        # ---- Residual connection ----
        out = x_agg + residual

        return out

    def get_frame_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the learned frame importance weights for analysis.
        Useful for visualizing which frames the model focuses on.

        Args:
            x: [B, F, N, D] (only the frame dimension is used)
        Returns:
            frame_weights: [n_frames] normalized attention weights
        """
        n_frames = x.shape[1]
        w = self.frame_weights[:n_frames]
        return F.softmax(w.squeeze(-1), dim=0)


if __name__ == "__main__":
    # Quick test
    B, F, N, D = 4, 10, 192, 768
    encoder = TemporalEventEncoder(dim=D, num_layers=2, num_heads=8)
    x = torch.randn(B, F, N, D)
    out = encoder(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Frame weights: {encoder.get_frame_attention_weights(x)}")
    print(f"Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

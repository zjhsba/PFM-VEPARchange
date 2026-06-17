"""
Multi-Modal Contrastive Loss for RGB-Event Pedestrian Attribute Recognition

Implements InfoNCE contrastive loss between RGB and Event modalities to explicitly
align their feature spaces. This encourages the model to learn modality-invariant
representations that are useful for downstream attribute recognition.

Architecture follows SimCLR-style contrastive learning:
  1. Global pooling of patch-level features → [B, D]
  2. Small projection MLP head → [B, proj_out_dim]
  3. L2 normalization
  4. InfoNCE loss with symmetric formulation

Positive pairs: (rgb_i, event_i) from the same pedestrian sample
Negative pairs: (rgb_i, event_j) for all i ≠ j in the batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss between RGB and Event global features.

    For each sample in the batch:
      - Positive: the paired (rgb, event) from the same pedestrian
      - Negatives: all other pairs in the batch

    Uses symmetric loss: (rgb→event + event→rgb) / 2

    Args:
        temperature: Softmax temperature (lower = sharper distribution)
        proj_hidden_dim: Hidden dimension of the projection MLP
        proj_out_dim: Output dimension of the projection head
        input_dim: Input feature dimension (default 768 for ViT-B)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        input_dim: int = 768,
    ):
        super().__init__()
        self.temperature = temperature

        # Projection head for RGB features
        self.rgb_proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

        # Projection head for Event features (separate weights)
        self.event_proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        rgb_features: torch.Tensor,
        event_features: torch.Tensor,
        return_z: bool = False,
    ):
        """
        Compute multi-modal contrastive loss.

        Args:
            rgb_features:   [B, N, D] RGB patch features (pre-fusion)
            event_features: [B, N, D] Event patch features (pre-fusion)
            return_z:       If True, also return (rgb_z, event_z) for monitoring

        Returns:
            contrastive_loss: scalar tensor, or (loss, rgb_z, event_z) if return_z
        """
        B = rgb_features.shape[0]

        # ---- Global pooling: mean over spatial patches ----
        if rgb_features.dim() == 3:
            rgb_global = rgb_features.mean(dim=1)    # [B, D]
        else:
            rgb_global = rgb_features

        if event_features.dim() == 3:
            event_global = event_features.mean(dim=1)  # [B, D]
        else:
            event_global = event_features

        # ---- Project through MLP heads ----
        rgb_z = self.rgb_proj(rgb_global)      # [B, out_dim]
        event_z = self.event_proj(event_global)  # [B, out_dim]

        # ---- L2 normalize ----
        rgb_z = F.normalize(rgb_z, dim=-1)
        event_z = F.normalize(event_z, dim=-1)

        # ---- Compute similarity matrix ----
        # logits[i, j] = similarity between rgb_i and event_j
        logits = torch.matmul(rgb_z, event_z.T) / self.temperature  # [B, B]

        # ---- InfoNCE loss ----
        # Diagonal is positive pairs
        labels = torch.arange(B, device=rgb_features.device)

        # Symmetric: rgb→event + event→rgb
        loss_rgb_to_event = F.cross_entropy(logits, labels)
        loss_event_to_rgb = F.cross_entropy(logits.T, labels)

        loss = (loss_rgb_to_event + loss_event_to_rgb) / 2.0

        if return_z:
            return loss, rgb_z, event_z
        return loss

    def get_similarity_matrix(
        self,
        rgb_features: torch.Tensor,
        event_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the similarity matrix for analysis/visualization.
        Returns [B, B] matrix where entry [i, j] = sim(rgb_i, event_j).
        """
        rgb_global = rgb_features.mean(dim=1) if rgb_features.dim() == 3 else rgb_features
        event_global = event_features.mean(dim=1) if event_features.dim() == 3 else event_features

        rgb_z = F.normalize(self.rgb_proj(rgb_global), dim=-1)
        event_z = F.normalize(self.event_proj(event_global), dim=-1)

        return torch.matmul(rgb_z, event_z.T)


class MultiModalContrastiveLossV2(nn.Module):
    """
    Extended contrastive loss with additional patch-level contrast.

    In addition to the global (image-level) contrast, this version also
    applies a lightweight patch-level contrast to encourage local alignment
    between RGB and Event modalities.

    L_total = L_global + λ * L_patch
    """

    def __init__(
        self,
        temperature: float = 0.07,
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        input_dim: int = 768,
        patch_lambda: float = 0.1,
        num_patch_samples: int = 16,
    ):
        super().__init__()
        self.temperature = temperature
        self.patch_lambda = patch_lambda
        self.num_patch_samples = num_patch_samples

        # Global projection heads
        self.rgb_global_proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )
        self.event_global_proj = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden_dim, proj_out_dim),
        )

        # Shared patch-level projection (lighter)
        self.patch_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _global_contrastive_loss(
        self,
        rgb_global: torch.Tensor,
        event_global: torch.Tensor,
    ) -> torch.Tensor:
        """Image-level contrastive loss."""
        B = rgb_global.shape[0]
        rgb_z = F.normalize(self.rgb_global_proj(rgb_global), dim=-1)
        event_z = F.normalize(self.event_global_proj(event_global), dim=-1)
        logits = torch.matmul(rgb_z, event_z.T) / self.temperature
        labels = torch.arange(B, device=rgb_global.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0
        return loss

    def _patch_contrastive_loss(
        self,
        rgb_patches: torch.Tensor,
        event_patches: torch.Tensor,
    ) -> torch.Tensor:
        """
        Patch-level contrastive loss.
        Randomly samples patches and aligns corresponding positions.
        """
        B, N, D = rgb_patches.shape

        # Randomly sample patch positions
        if N > self.num_patch_samples:
            indices = torch.randperm(N, device=rgb_patches.device)[:self.num_patch_samples]
        else:
            indices = torch.arange(N, device=rgb_patches.device)

        rgb_sampled = rgb_patches[:, indices, :]    # [B, K, D]
        event_sampled = event_patches[:, indices, :]  # [B, K, D]

        # Reshape: [B*K, D]
        rgb_flat = rgb_sampled.reshape(-1, D)
        event_flat = event_sampled.reshape(-1, D)

        # Project
        rgb_p = F.normalize(self.patch_proj(rgb_flat), dim=-1)
        event_p = F.normalize(self.patch_proj(event_flat), dim=-1)

        # InfoNCE (positive = same position across modalities)
        K = len(indices)
        logits = torch.matmul(rgb_p, event_p.T) / self.temperature  # [B*K, B*K]
        labels = torch.arange(B * K, device=rgb_patches.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

        return loss

    def forward(
        self,
        rgb_features: torch.Tensor,
        event_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            rgb_features:   [B, N, D]
            event_features: [B, N, D]
        Returns:
            total contrastive loss
        """
        # Global contrast
        rgb_global = rgb_features.mean(dim=1)
        event_global = event_features.mean(dim=1)
        loss_global = self._global_contrastive_loss(rgb_global, event_global)

        # Patch contrast
        loss_patch = self._patch_contrastive_loss(rgb_features, event_features)

        return loss_global + self.patch_lambda * loss_patch


if __name__ == "__main__":
    # Quick test
    B, N, D = 8, 192, 768
    rgb = torch.randn(B, N, D)
    event = torch.randn(B, N, D)

    # Test basic version
    loss_fn = MultiModalContrastiveLoss(temperature=0.07)
    loss = loss_fn(rgb, event)
    print(f"Basic contrastive loss: {loss.item():.4f}")

    # Test V2 version
    loss_fn_v2 = MultiModalContrastiveLossV2(temperature=0.07)
    loss_v2 = loss_fn_v2(rgb, event)
    print(f"V2 contrastive loss: {loss_v2.item():.4f}")

    # Verify positive pair has high similarity
    sim_matrix = loss_fn.get_similarity_matrix(rgb, event)
    diag_sim = sim_matrix.diagonal().mean().item()
    off_diag_sim = (sim_matrix.sum() - sim_matrix.diagonal().sum()) / (B * (B - 1))
    print(f"Avg positive similarity: {diag_sim:.4f}")
    print(f"Avg negative similarity: {off_diag_sim:.4f}")
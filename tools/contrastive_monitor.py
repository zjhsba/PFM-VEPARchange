"""
Contrastive Learning Monitor — tracks InfoNCE health signals, gradient flow,
and temporal encoder activation to verify that new modules are "eating well".

Tracks per step and per epoch:
  - contrastive loss
  - positive / negative similarity (InfoNCE signal quality)
  - gradient norms for temporal encoder & contrastive projection layers
  - temporal frame-weight distribution entropy
  - activation statistics (mean/std) of projection outputs

Writes a CSV log for post-hoc analysis and prints a concise epoch summary.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


class ContrastiveMonitor:
    """Non-intrusive monitor that records diagnostic signals during training."""

    def __init__(
        self,
        model: torch.nn.Module,
        csv_path: str,
        enabled: bool = True,
    ):
        self.model = model
        self.enabled = enabled
        self.csv_path = csv_path

        # ---- per-step accumulators (reset each epoch) ----
        self.contrast_losses: list[float] = []
        self.pos_sims: list[float] = []          # diagonal of sim matrix
        self.neg_sims: list[float] = []          # off-diagonal mean of sim matrix
        self.frame_entropies: list[float] = []   # entropy of learnable frame weights
        self.proj_rgb_stds: list[float] = []     # std of rgb projection output
        self.proj_event_stds: list[float] = []   # std of event projection output

        # ---- per-epoch summary (written to CSV) ----
        self.epoch_history: list[dict] = []

        # ---- hook handles ----
        self._handles = []

        if self.enabled:
            self._csv_init()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self,
        contrast_loss: float,
        rgb_z=None,
        event_z=None,
        temporal_encoder=None,
    ):
        """Call after each training step from batch_engine."""
        if not self.enabled:
            return

        self.contrast_losses.append(contrast_loss)

        # ---- positive / negative similarity ----
        if rgb_z is not None and event_z is not None:
            pos, neg = self._compute_similarity_stats(rgb_z, event_z)
            self.pos_sims.append(pos)
            self.neg_sims.append(neg)

            # activation statistics
            self.proj_rgb_stds.append(rgb_z.std(dim=-1).mean().item())
            self.proj_event_stds.append(event_z.std(dim=-1).mean().item())

        # ---- frame weight entropy ----
        if temporal_encoder is not None:
            self.frame_entropies.append(
                self._compute_frame_entropy(temporal_encoder)
            )

    def record_grad_norms(self, contrastive_criterion=None):
        """Call after loss.backward() to snapshot gradient norms.

        Walks the model's named_parameters to find temporal + contrastive params.
        """
        if not self.enabled:
            return {}

        norms = {}
        module = self.model.module if hasattr(self.model, 'module') else self.model

        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            if 'temporal' in name or 'contrast' in name or 'proj' in name:
                g_norm = param.grad.norm(2).item()
                norms[name] = g_norm

        return norms

    def epoch_summary(self, grad_norms=None) -> dict:
        """Return a summary dict for the current epoch and reset accumulators."""
        if not self.enabled or not self.contrast_losses:
            return {}

        summary = {
            'contrast_loss': float(np.mean(self.contrast_losses)),
            'pos_sim': float(np.mean(self.pos_sims)) if self.pos_sims else float('nan'),
            'neg_sim': float(np.mean(self.neg_sims)) if self.neg_sims else float('nan'),
            'snr': float(np.mean(self.pos_sims) / (np.mean(self.neg_sims) + 1e-8))
                   if self.pos_sims and self.neg_sims else float('nan'),
            'proj_rgb_std': float(np.mean(self.proj_rgb_stds)) if self.proj_rgb_stds else float('nan'),
            'proj_event_std': float(np.mean(self.proj_event_stds)) if self.proj_event_stds else float('nan'),
            'frame_entropy': float(np.mean(self.frame_entropies)) if self.frame_entropies else float('nan'),
        }

        if grad_norms:
            summary['grad_temporal_mean'] = float(np.mean([
                v for k, v in grad_norms.items() if 'temporal' in k
            ])) if any('temporal' in k for k in grad_norms) else float('nan')
            summary['grad_contrast_proj_mean'] = float(np.mean([
                v for k, v in grad_norms.items() if 'proj' in k or 'contrast' in k
            ])) if any('proj' in k or 'contrast' in k for k in grad_norms) else float('nan')

        self.epoch_history.append(summary)
        self._csv_append(summary)
        self._reset_epoch()

        return summary

    def print_epoch_summary(self, summary: dict):
        if not summary:
            return
        print(
            f'  [Contrastive] loss={summary["contrast_loss"]:.4f}  '
            f'pos_sim={summary["pos_sim"]:.4f}  neg_sim={summary["neg_sim"]:.4f}  '
            f'SNR={summary["snr"]:.2f}  '
            f'|z_rgb|={summary["proj_rgb_std"]:.3f}  |z_event|={summary["proj_event_std"]:.3f}  '
            f'H(frame)={summary["frame_entropy"]:.3f}'
        )
        if 'grad_temporal_mean' in summary and not math.isnan(summary['grad_temporal_mean']):
            print(
                f'  [Grad Norms] temporal={summary["grad_temporal_mean"]:.4f}  '
                f'proj={summary["grad_contrast_proj_mean"]:.4f}'
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_similarity_stats(
        self, rgb_z: torch.Tensor, event_z: torch.Tensor
    ) -> tuple[float, float]:
        """Compute pos/neg similarity from L2-normalised projection outputs."""
        with torch.no_grad():
            B = rgb_z.shape[0]
            sim = torch.matmul(rgb_z, event_z.T)  # [B, B]
            pos = sim.diagonal().mean().item()
            if B > 1:
                neg = (sim.sum() - sim.diagonal().sum()) / (B * (B - 1))
                neg = neg.item()
            else:
                neg = 0.0
        return pos, neg

    def _compute_frame_entropy(
        self, temporal_encoder: torch.nn.Module
    ) -> float:
        """Entropy of the softmax-normalised frame weights."""
        with torch.no_grad():
            w = temporal_encoder.frame_weights  # [max_frames, 1]
            w_soft = F.softmax(w.squeeze(-1), dim=0)
            # Entropy: -sum(p * log p), max for uniform = log(n)
            eps = 1e-8
            entropy = -(w_soft * torch.log(w_soft + eps)).sum().item()
            max_entropy = math.log(len(w_soft))
            # Normalise to [0, 1]
            return entropy / max_entropy if max_entropy > 0 else 0.0

    def _csv_init(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'contrast_loss',
                    'pos_sim', 'neg_sim', 'snr',
                    'proj_rgb_std', 'proj_event_std',
                    'frame_entropy',
                    'grad_temporal_mean', 'grad_contrast_proj_mean',
                ])

    def _csv_append(self, summary: dict):
        epoch = len(self.epoch_history)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                summary.get('contrast_loss', ''),
                summary.get('pos_sim', ''),
                summary.get('neg_sim', ''),
                summary.get('snr', ''),
                summary.get('proj_rgb_std', ''),
                summary.get('proj_event_std', ''),
                summary.get('frame_entropy', ''),
                summary.get('grad_temporal_mean', ''),
                summary.get('grad_contrast_proj_mean', ''),
            ])

    def _reset_epoch(self):
        self.contrast_losses.clear()
        self.pos_sims.clear()
        self.neg_sims.clear()
        self.frame_entropies.clear()
        self.proj_rgb_stds.clear()
        self.proj_event_stds.clear()

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

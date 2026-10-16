"""Small TS2Vec-style dilated TCN with optional DANN domain head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output.neg() * ctx.coefficient, None


def gradient_reverse(x: Tensor, coefficient: float) -> Tensor:
    return _GradientReverse.apply(x, coefficient)


class DilatedResidualBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _norm(x: Tensor, norm: nn.LayerNorm) -> Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self._norm(self.conv1(x), self.norm1))
        x = self.dropout(x)
        x = self._norm(self.conv2(x), self.norm2)
        return F.gelu(residual + self.dropout(x))


class TS2VecDANN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dilations: list[int], dropout: float, n_domains: int):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([DilatedResidualBlock(hidden_dim, d, dropout) for d in dilations])
        self.project = nn.Sequential(nn.LayerNorm(hidden_dim * 3), nn.Linear(hidden_dim * 3, output_dim))
        self.domain_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_domains)
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # input [B, T, F], temporal states [B, T, H], global vector [B, D]
        h = self.input(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)
        pooled = torch.cat([h[:, -1], h.mean(dim=1), h.max(dim=1).values], dim=-1)
        z = F.normalize(self.project(pooled), dim=-1)
        return h, z

    def forward(self, x: Tensor, grl_coefficient: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
        h, z = self.encode(x)
        return h, z, self.domain_head(gradient_reverse(z, grl_coefficient))


def augment_window(x: Tensor, point_mask_probability: float, contiguous_mask_probability: float) -> Tensor:
    """Mask-only views preserve financial chronology and use zero = robust median."""
    out = x.clone()
    if point_mask_probability:
        out = out.masked_fill(torch.rand_like(out) < point_mask_probability, 0.0)
    if contiguous_mask_probability:
        b, t, _ = out.shape
        length = max(1, int(round(t * contiguous_mask_probability)))
        starts = torch.randint(0, t - length + 1, (b,), device=x.device)
        for i, start in enumerate(starts.tolist()):
            out[i, start:start + length] = 0.0
    return out


def nt_xent(a: Tensor, b: Tensor, temperature: float = 0.2) -> Tensor:
    """Symmetric in-batch contrastive loss; stable for modest GPU batches."""
    logits = (F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T) / temperature
    labels = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def hierarchical_ssl_loss(h1: Tensor, z1: Tensor, h2: Tensor, z2: Tensor) -> Tensor:
    """Instance loss plus multi-scale temporal contrast on paired augmented views."""
    loss = nt_xent(z1, z2)
    levels = 0
    a, b = h1, h2
    while a.shape[1] >= 2:
        # At each time position contrast stocks within the batch.  This is the
        # temporal TS2Vec idea, but avoids a (B*T)^2 similarity matrix (which
        # would be needlessly large on an 8-GiB GPU).
        left = F.normalize(a, dim=-1).transpose(0, 1)   # [T, B, H]
        right = F.normalize(b, dim=-1).transpose(0, 1) # [T, B, H]
        logits = torch.bmm(left, right.transpose(1, 2)) / 0.2
        labels = torch.arange(a.shape[0], device=a.device).expand(a.shape[1], -1)
        temporal = 0.5 * (
            F.cross_entropy(logits.reshape(-1, a.shape[0]), labels.reshape(-1))
            + F.cross_entropy(logits.transpose(1, 2).reshape(-1, a.shape[0]), labels.reshape(-1))
        )
        loss = loss + temporal
        levels += 1
        if a.shape[1] == 2:
            break
        a = F.avg_pool1d(a.transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
        b = F.avg_pool1d(b.transpose(1, 2), 2, ceil_mode=True).transpose(1, 2)
    return loss / (levels + 1)


def dann_coefficient(progress: float, warmup: float, ramp_end: float, maximum: float) -> float:
    if progress <= warmup:
        return 0.0
    if progress >= ramp_end:
        return maximum
    fraction = (progress - warmup) / max(ramp_end - warmup, 1e-6)
    return maximum * (2.0 / (1.0 + math.exp(-10.0 * fraction)) - 1.0)


def coral_loss(z: Tensor, domains: Tensor, target_domain: int = 0) -> Tensor:
    """Conventional CORAL distance between target A-share and reference covariances.

    Batches are market-balanced by construction. Markets with fewer than two
    observations are skipped, although normal production batches never do so.
    """
    target = z[domains == target_domain]
    if len(target) < 2:
        return z.new_zeros(())
    target_cov = (target - target.mean(0)).T @ (target - target.mean(0)) / (len(target) - 1)
    total, count, d = z.new_zeros(()), 0, z.shape[-1]
    for domain in domains.unique().tolist():
        if domain == target_domain:
            continue
        source = z[domains == domain]
        if len(source) < 2:
            continue
        source_cov = (source - source.mean(0)).T @ (source - source.mean(0)) / (len(source) - 1)
        total = total + (source_cov - target_cov).pow(2).sum() / (4.0 * d * d)
        count += 1
    return total / max(count, 1)

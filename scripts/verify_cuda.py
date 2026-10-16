#!/usr/bin/env python3
"""Fail fast if this training environment cannot see NVIDIA CUDA."""
from __future__ import annotations

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Do not begin this training run on CPU.")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM (GiB): {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")

"""Device + autocast resolution, shared by training and evaluation.

Imported lazily by torch-using modules so ``config`` stays torch-free.
"""
from __future__ import annotations

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# bf16 autocast on Ampere+; None (fp32) on CPU or older GPUs.
AMP_DTYPE = (torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported())
             else None)

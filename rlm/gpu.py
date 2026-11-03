"""GPU workarounds shared by every script that generates or trains.

On the course DGX (H200 MIG slice 3g.71gb, torch 2.11, transformers 5.17) PyTorch picks the cuDNN
backend for ``scaled_dot_product_attention`` and decoding becomes 3-5x slower than with the
other SDPA backends: 0.46 s per step instead of 0.09 s for Qwen3-0.6B with a batch of 32 and the
2600-token AI Act prompt (EXPERIMENTS.md, 2026-10-30). Disabling it costs nothing elsewhere.
"""

from __future__ import annotations


def disable_cudnn_attention() -> None:
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)

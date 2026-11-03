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


_SHARED: dict[str, tuple] = {}


def shared_causal_lm(model_id: str):
    """(tokenizer, model) for ``model_id``, loaded once per process in bf16 on GPU.

    Phases 2, 3 and 4 use the same instruction model; on a 16 GB GPU two copies of Qwen3-4B do
    not fit next to the embedding model, so the API shares one instance between them.
    """
    if model_id not in _SHARED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device)
        _SHARED[model_id] = (tokenizer, model.eval())
    return _SHARED[model_id]

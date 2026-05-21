"""PyTorch baselines for attention microbenchmarks."""

from __future__ import annotations

from typing import Any


def decode_attention(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    *,
    scale: float | None = None,
    out: Any | None = None,
) -> Any:
    """Return one-token scaled dot-product attention over a contiguous KV cache.

    Shapes:
    - query: ``(num_heads, head_dim)``
    - key_cache/value_cache: ``(seq_len, num_heads, head_dim)``
    - output: ``(num_heads, head_dim)``
    """

    _require_decode_inputs(query, key_cache, value_cache)
    effective_scale = scale if scale is not None else query.shape[-1] ** -0.5
    scores = _einsum("hd,shd->hs", query.float(), key_cache.float()) * effective_scale
    probs = scores.softmax(dim=-1).to(dtype=query.dtype)
    result = _einsum("hs,shd->hd", probs, value_cache)
    if out is None:
        return result
    return out.copy_(result)


def _require_decode_inputs(query: Any, key_cache: Any, value_cache: Any) -> None:
    if getattr(query, "ndim", None) != 2:
        raise ValueError(f"query must be 2D, got ndim={getattr(query, 'ndim', None)}")
    if getattr(key_cache, "ndim", None) != 3:
        raise ValueError(f"key_cache must be 3D, got ndim={getattr(key_cache, 'ndim', None)}")
    if getattr(value_cache, "ndim", None) != 3:
        raise ValueError(
            f"value_cache must be 3D, got ndim={getattr(value_cache, 'ndim', None)}"
        )
    if getattr(key_cache, "shape", None) != getattr(value_cache, "shape", None):
        raise ValueError(
            f"key/value cache shape mismatch: {key_cache.shape} != {value_cache.shape}"
        )
    _, num_heads, head_dim = key_cache.shape
    if query.shape != (num_heads, head_dim):
        raise ValueError(
            "query shape must match cache heads and head_dim: "
            f"{query.shape} != ({num_heads}, {head_dim})"
        )
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise ValueError("query, key_cache, and value_cache must have the same dtype")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query, key_cache, and value_cache must be on the same device")


def _einsum(equation: str, left: Any, right: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Attention baseline requires torch to be installed.") from exc

    return torch.einsum(equation, left, right)

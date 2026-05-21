"""Accounting models for decode attention microbenchmarks."""

from __future__ import annotations


def decode_attention_memory_traffic_bytes(
    *,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
) -> int:
    """Estimate fused HBM traffic for one-token decode attention.

    The model counts one query read, K/V cache reads, and one output write. It
    intentionally omits intermediate score/probability tensors so the estimate
    describes the fused kernel target rather than a framework baseline.
    """

    _require_decode_shape(seq_len=seq_len, num_heads=num_heads, head_dim=head_dim)
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    query_values = num_heads * head_dim
    cache_values = seq_len * num_heads * head_dim
    output_values = num_heads * head_dim
    return (query_values + 2 * cache_values + output_values) * dtype_size


def decode_attention_flop_count(*, seq_len: int, num_heads: int, head_dim: int) -> int:
    """Estimate scalar work for one-token scaled dot-product decode attention."""

    _require_decode_shape(seq_len=seq_len, num_heads=num_heads, head_dim=head_dim)
    qk_dot = seq_len * (2 * head_dim)
    softmax = 5 * seq_len
    value_mix = seq_len * (2 * head_dim)
    return num_heads * (qk_dot + softmax + value_mix)


def _require_decode_shape(*, seq_len: int, num_heads: int, head_dim: int) -> None:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")

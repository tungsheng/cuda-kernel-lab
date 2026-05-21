"""Accounting models for synthetic decode-step benchmarks."""

from __future__ import annotations


def decode_step_memory_traffic_bytes(
    *,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
) -> int:
    """Estimate HBM traffic for the synthetic one-token decode step."""

    _require_decode_step_shape(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    attention_dim = num_heads * head_dim
    rmsnorm_values = 3 * batch_size * hidden_dim + hidden_dim
    q_projection_values = batch_size * hidden_dim + hidden_dim * attention_dim + (
        batch_size * attention_dim
    )
    cache_values = 2 * batch_size * seq_len * attention_dim
    attention_values = batch_size * attention_dim + cache_values + batch_size * attention_dim
    ffn_projection_values = 2 * (
        batch_size * hidden_dim + hidden_dim * intermediate_dim + batch_size * intermediate_dim
    )
    swiglu_values = 3 * batch_size * intermediate_dim
    output_values = 3 * batch_size * attention_dim
    total_values = (
        rmsnorm_values
        + q_projection_values
        + attention_values
        + ffn_projection_values
        + swiglu_values
        + output_values
    )
    return total_values * dtype_size


def decode_step_flop_count(
    *,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> int:
    """Estimate scalar work for the synthetic one-token decode step."""

    _require_decode_step_shape(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    attention_dim = num_heads * head_dim
    rmsnorm_flops = batch_size * (7 + 4 * hidden_dim)
    q_projection_flops = 2 * batch_size * hidden_dim * attention_dim
    attention_flops = batch_size * num_heads * (
        seq_len * (2 * head_dim) + 5 * seq_len + seq_len * (2 * head_dim)
    )
    ffn_projection_flops = 4 * batch_size * hidden_dim * intermediate_dim
    swiglu_flops = 5 * batch_size * intermediate_dim
    output_flops = batch_size * attention_dim
    return (
        rmsnorm_flops
        + q_projection_flops
        + attention_flops
        + ffn_projection_flops
        + swiglu_flops
        + output_flops
    )


def _require_decode_step_shape(
    *,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if intermediate_dim <= 0:
        raise ValueError("intermediate_dim must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if intermediate_dim < num_heads * head_dim:
        raise ValueError("intermediate_dim must be at least num_heads * head_dim")

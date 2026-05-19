"""Accounting models for row-wise normalization primitives."""

from __future__ import annotations


def memory_traffic_bytes(
    op_name: str,
    *,
    rows: int,
    cols: int,
    dtype_size: int,
) -> int:
    """Estimate fused HBM traffic for row-wise normalization.

    RMSNorm reads input and weight, then writes output. LayerNorm reads input,
    weight, and bias, then writes output. This deliberately ignores cache reuse
    of the parameter vector so comparisons stay conservative and easy to audit.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    numel = rows * cols
    if op_name == "rmsnorm":
        return 3 * numel * dtype_size
    if op_name == "layernorm":
        return 4 * numel * dtype_size
    raise ValueError(f"unknown normalization primitive: {op_name}")


def flop_count(op_name: str, *, rows: int, cols: int) -> int:
    """Estimate scalar operations for row-wise normalization."""

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    if op_name == "rmsnorm":
        # square, sum reduction, divide, rsqrt, scale, weight multiply
        return rows * (max(cols - 1, 0) + 4 * cols)
    if op_name == "layernorm":
        # mean reduction, variance reduction, normalize, affine scale/bias
        return rows * (2 * max(cols - 1, 0) + 6 * cols)
    raise ValueError(f"unknown normalization primitive: {op_name}")

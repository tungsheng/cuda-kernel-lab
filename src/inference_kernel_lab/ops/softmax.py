"""Accounting models for row-wise softmax."""

from __future__ import annotations


def memory_traffic_bytes(
    *,
    rows: int,
    cols: int,
    dtype_size: int,
    model: str = "fused",
) -> int:
    """Estimate high bandwidth memory traffic for row-wise softmax.

    The fused model is the idealized lower bound: read input once and write
    output once. The naive model represents a two-kernel implementation that
    writes and rereads an intermediate tensor.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    numel = rows * cols
    if model == "fused":
        return 2 * numel * dtype_size
    if model == "naive":
        return 4 * numel * dtype_size
    raise ValueError(f"unknown softmax traffic model: {model}")


def flop_count(*, rows: int, cols: int) -> int:
    """Estimate scalar operations for row-wise softmax.

    This is a simple accounting model for benchmark reporting, not a substitute
    for profiler-level instruction analysis.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    # Per row: max reduction, subtract, exp, sum reduction, divide.
    return rows * (2 * max(cols - 1, 0) + 3 * cols)

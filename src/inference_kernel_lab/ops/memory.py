"""Accounting models for memory-bandwidth primitives."""

from __future__ import annotations


def memory_traffic_bytes(op_name: str, *, numel: int, dtype_size: int) -> int:
    """Estimate high bandwidth memory traffic for a memory primitive.

    The model counts full tensor reads and writes. Reduction is approximated as
    one full input read plus one scalar output write.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    if op_name == "vector_add":
        return 3 * numel * dtype_size
    if op_name in {"copy", "scale"}:
        return 2 * numel * dtype_size
    if op_name == "reduction_sum":
        return (numel + 1) * dtype_size
    raise ValueError(f"unknown memory primitive: {op_name}")


def flop_count(op_name: str, *, numel: int) -> int:
    """Estimate scalar floating point operations for a memory primitive."""

    if numel < 0:
        raise ValueError("numel must be non-negative")

    if op_name in {"vector_add", "scale"}:
        return numel
    if op_name == "copy":
        return 0
    if op_name == "reduction_sum":
        return max(numel - 1, 0)
    raise ValueError(f"unknown memory primitive: {op_name}")

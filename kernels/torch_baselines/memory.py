"""PyTorch baselines for memory-bandwidth primitives."""

from __future__ import annotations

from typing import Any


def vector_add(a: Any, b: Any) -> Any:
    """Return elementwise a + b."""

    _require_same_shape(a, b)
    return a + b


def copy(x: Any) -> Any:
    """Return a materialized copy of x."""

    return x.clone()


def scale(x: Any, alpha: float) -> Any:
    """Return x scaled by alpha."""

    return x * alpha


def reduction_sum(x: Any, dim: int | None = None) -> Any:
    """Return the sum of x, optionally along one dimension."""

    return x.sum() if dim is None else x.sum(dim=dim)


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


def _require_same_shape(a: Any, b: Any) -> None:
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        left = getattr(a, "shape", None)
        right = getattr(b, "shape", None)
        raise ValueError(f"shape mismatch: {left} != {right}")

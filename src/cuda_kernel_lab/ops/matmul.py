"""Accounting models for matrix multiplication."""

from __future__ import annotations


def memory_traffic_bytes(*, m: int, n: int, k: int, dtype_size: int) -> int:
    """Estimate lower-bound HBM traffic for one matmul."""

    _validate_shape(m=m, n=n, k=k)
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")
    return (m * k + k * n + m * n) * dtype_size


def flop_count(*, m: int, n: int, k: int) -> int:
    """Estimate multiply-add FLOPs for matmul."""

    _validate_shape(m=m, n=n, k=k)
    return 2 * m * n * k


def _validate_shape(*, m: int, n: int, k: int) -> None:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")

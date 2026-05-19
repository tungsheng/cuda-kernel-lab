"""Accounting models for SwiGLU elementwise fusion."""

from __future__ import annotations


def memory_traffic_bytes(*, numel: int, dtype_size: int) -> int:
    """Estimate HBM traffic for fused SwiGLU.

    The fused model reads the gate and up tensors once, then writes one output
    tensor. Intermediate sigmoid/SwiLU values stay inside the kernel.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")
    return 3 * numel * dtype_size


def flop_count(*, numel: int) -> int:
    """Estimate scalar operations for fused SwiGLU.

    Per element this counts exp, add, reciprocal, gate multiply, and up multiply.
    This is an accounting model for reports, not an instruction-level profile.
    """

    if numel < 0:
        raise ValueError("numel must be non-negative")
    return 5 * numel

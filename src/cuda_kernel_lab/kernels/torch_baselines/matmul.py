"""PyTorch baseline for matrix multiplication."""

from __future__ import annotations

from typing import Any


def matmul(a: Any, b: Any, *, out: Any | None = None) -> Any:
    """Return a @ b."""

    _require_matrices(a, b)
    result = a @ b
    if out is None:
        return result
    return out.copy_(result)


def _require_matrices(a: Any, b: Any) -> None:
    if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
        left = getattr(a, "ndim", None)
        right = getattr(b, "ndim", None)
        raise ValueError(f"matmul requires two 2D tensors, got ndim={left} and ndim={right}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a.shape} cannot multiply {b.shape}")

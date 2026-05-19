"""PyTorch baselines for memory-bandwidth primitives."""

from __future__ import annotations

from typing import Any


def vector_add(a: Any, b: Any, *, out: Any | None = None) -> Any:
    """Return elementwise a + b."""

    _require_same_shape(a, b)
    if out is None:
        return a + b
    return _torch_add(a, b, out)


def copy(x: Any, *, out: Any | None = None) -> Any:
    """Return a materialized copy of x."""

    if out is None:
        return x.clone()
    return out.copy_(x)


def scale(x: Any, alpha: float, *, out: Any | None = None) -> Any:
    """Return x scaled by alpha."""

    if out is None:
        return x * alpha
    return _torch_mul(x, alpha, out)


def reduction_sum(x: Any, dim: int | None = None) -> Any:
    """Return the sum of x, optionally along one dimension."""

    return x.sum() if dim is None else x.sum(dim=dim)


def _require_same_shape(a: Any, b: Any) -> None:
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        left = getattr(a, "shape", None)
        right = getattr(b, "shape", None)
        raise ValueError(f"shape mismatch: {left} != {right}")


def _torch_add(a: Any, b: Any, out: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch memory baseline requires torch to be installed.") from exc

    return torch.add(a, b, out=out)


def _torch_mul(x: Any, alpha: float, out: Any) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch memory baseline requires torch to be installed.") from exc

    return torch.mul(x, alpha, out=out)

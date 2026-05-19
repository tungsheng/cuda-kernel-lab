"""PyTorch baseline for row-wise softmax."""

from __future__ import annotations

from typing import Any


def softmax(x: Any, *, out: Any | None = None) -> Any:
    """Return row-wise softmax over the last dimension of a 2D tensor."""

    _require_2d(x)
    result = x.softmax(dim=-1)
    if out is None:
        return result
    return out.copy_(result)


def _require_2d(x: Any) -> None:
    if getattr(x, "ndim", None) != 2:
        ndim = getattr(x, "ndim", None)
        raise ValueError(f"row-wise softmax requires a 2D tensor, got ndim={ndim}")

"""PyTorch baseline for SwiGLU elementwise fusion."""

from __future__ import annotations

from typing import Any


def swiglu(gate: Any, up: Any, *, out: Any | None = None) -> Any:
    """Return SiLU(gate) * up."""

    _require_same_shape(gate, up)
    result = gate * gate.sigmoid() * up
    if out is None:
        return result
    return out.copy_(result)


def _require_same_shape(a: Any, b: Any) -> None:
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        left = getattr(a, "shape", None)
        right = getattr(b, "shape", None)
        raise ValueError(f"shape mismatch: {left} != {right}")

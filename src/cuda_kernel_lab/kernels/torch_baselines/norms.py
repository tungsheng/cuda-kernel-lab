"""PyTorch baselines for row-wise normalization kernels."""

from __future__ import annotations

from typing import Any


def rmsnorm(x: Any, weight: Any, *, eps: float = 1e-6, out: Any | None = None) -> Any:
    """Return row-wise RMSNorm over the last dimension."""

    _require_matrix_and_weight(x, weight)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * variance.add(eps).rsqrt().to(dtype=x.dtype)
    result = normalized * weight
    if out is None:
        return result
    return out.copy_(result)


def layernorm(
    x: Any,
    weight: Any,
    bias: Any,
    *,
    eps: float = 1e-5,
    out: Any | None = None,
) -> Any:
    """Return row-wise LayerNorm over the last dimension."""

    _require_matrix_and_weight(x, weight)
    if getattr(bias, "shape", None) != getattr(weight, "shape", None):
        raise ValueError(f"bias shape must match weight shape: {bias.shape} != {weight.shape}")
    result = _layer_norm(x, weight, bias, eps)
    if out is None:
        return result
    return out.copy_(result)


def _require_matrix_and_weight(x: Any, weight: Any) -> None:
    if getattr(x, "ndim", None) != 2:
        ndim = getattr(x, "ndim", None)
        raise ValueError(f"normalization requires a 2D input tensor, got ndim={ndim}")
    if getattr(weight, "ndim", None) != 1:
        ndim = getattr(weight, "ndim", None)
        raise ValueError(f"normalization weight must be 1D, got ndim={ndim}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight length must match input columns: {weight.shape[0]} != {x.shape[-1]}"
        )


def _layer_norm(x: Any, weight: Any, bias: Any, eps: float) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("LayerNorm baseline requires torch to be installed.") from exc

    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)

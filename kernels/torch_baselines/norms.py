"""PyTorch baselines for inference-time normalization kernels."""

from __future__ import annotations

from typing import Any


def rmsnorm(x: Any, weight: Any, *, eps: float = 1e-6) -> Any:
    """Return row-wise RMSNorm over the last dimension."""

    _require_matrix_and_weight(x, weight)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x * variance.add(eps).rsqrt().to(dtype=x.dtype)
    return normalized * weight


def layernorm(x: Any, weight: Any, bias: Any, *, eps: float = 1e-5) -> Any:
    """Return row-wise LayerNorm over the last dimension."""

    _require_matrix_and_weight(x, weight)
    if getattr(bias, "shape", None) != getattr(weight, "shape", None):
        raise ValueError(f"bias shape must match weight shape: {bias.shape} != {weight.shape}")
    return _layer_norm(x, weight, bias, eps)


def memory_traffic_bytes(
    op_name: str,
    *,
    rows: int,
    cols: int,
    dtype_size: int,
) -> int:
    """Estimate fused HBM traffic for row-wise normalization.

    RMSNorm reads input and weight, then writes output. LayerNorm reads input,
    weight, and bias, then writes output. This deliberately ignores cache reuse
    of the parameter vector so comparisons stay conservative and easy to audit.
    """

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if dtype_size <= 0:
        raise ValueError("dtype_size must be positive")

    numel = rows * cols
    if op_name == "rmsnorm":
        return 3 * numel * dtype_size
    if op_name == "layernorm":
        return 4 * numel * dtype_size
    raise ValueError(f"unknown normalization primitive: {op_name}")


def flop_count(op_name: str, *, rows: int, cols: int) -> int:
    """Estimate scalar operations for row-wise normalization."""

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    if op_name == "rmsnorm":
        # square, sum reduction, divide, rsqrt, scale, weight multiply
        return rows * (max(cols - 1, 0) + 4 * cols)
    if op_name == "layernorm":
        # mean reduction, variance reduction, normalize, affine scale/bias
        return rows * (2 * max(cols - 1, 0) + 6 * cols)
    raise ValueError(f"unknown normalization primitive: {op_name}")


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

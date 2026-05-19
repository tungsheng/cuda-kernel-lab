"""Triton row-wise RMSNorm and LayerNorm forward kernels."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:
    torch = None
    triton = None
    tl = None


MAX_BLOCK_SIZE = 131_072


def is_available() -> bool:
    """Return true when Triton and a CUDA-capable PyTorch runtime are available."""

    return bool(torch is not None and triton is not None and torch.cuda.is_available())


def rmsnorm(x: Any, weight: Any, *, eps: float = 1e-6, out: Any | None = None) -> Any:
    """Return row-wise RMSNorm over the last dimension using one Triton kernel."""

    _require_norm_inputs(x, weight)
    rows, cols = x.shape
    block_size = _block_size(cols)
    out = torch.empty_like(x) if out is None else out
    _require_norm_output(x, out)
    _rmsnorm_kernel[(rows,)](
        x,
        weight,
        out,
        x.stride(0),
        out.stride(0),
        cols,
        eps,
        block_size,
        num_warps=_num_warps(block_size),
    )
    return out


def layernorm(
    x: Any,
    weight: Any,
    bias: Any,
    *,
    eps: float = 1e-5,
    out: Any | None = None,
) -> Any:
    """Return row-wise LayerNorm over the last dimension using one Triton kernel."""

    _require_norm_inputs(x, weight, bias)
    rows, cols = x.shape
    block_size = _block_size(cols)
    out = torch.empty_like(x) if out is None else out
    _require_norm_output(x, out)
    _layernorm_kernel[(rows,)](
        x,
        weight,
        bias,
        out,
        x.stride(0),
        out.stride(0),
        cols,
        eps,
        block_size,
        num_warps=_num_warps(block_size),
    )
    return out


def _require_norm_inputs(x: Any, weight: Any, bias: Any | None = None) -> None:
    if torch is None or triton is None:
        raise RuntimeError("Triton normalization kernels require torch and triton.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton normalization kernels require a CUDA-capable PyTorch runtime.")
    if not getattr(x, "is_cuda", False):
        raise RuntimeError("Triton normalization kernels require CUDA tensors.")
    if not getattr(weight, "is_cuda", False):
        raise RuntimeError("Triton normalization weights must be CUDA tensors.")
    if bias is not None and not getattr(bias, "is_cuda", False):
        raise RuntimeError("Triton LayerNorm bias must be a CUDA tensor.")
    if getattr(x, "ndim", None) != 2:
        ndim = getattr(x, "ndim", None)
        raise ValueError(f"normalization requires a 2D input tensor, got ndim={ndim}")
    if getattr(weight, "ndim", None) != 1:
        ndim = getattr(weight, "ndim", None)
        raise ValueError(f"normalization weight must be 1D, got ndim={ndim}")
    if bias is not None and getattr(bias, "shape", None) != getattr(weight, "shape", None):
        raise ValueError(f"bias shape must match weight shape: {bias.shape} != {weight.shape}")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight length must match input columns: {weight.shape[0]} != {x.shape[-1]}"
        )
    if x.stride(1) != 1:
        raise ValueError("Triton normalization requires the last dimension to be contiguous.")
    if weight.stride(0) != 1:
        raise ValueError("Triton normalization requires contiguous weights.")
    if bias is not None and bias.stride(0) != 1:
        raise ValueError("Triton LayerNorm requires a contiguous bias.")


def _require_norm_output(x: Any, out: Any) -> None:
    if getattr(out, "shape", None) != getattr(x, "shape", None):
        raise ValueError(f"output shape must match input shape: {out.shape} != {x.shape}")
    if not getattr(out, "is_cuda", False):
        raise RuntimeError("Triton normalization output must be a CUDA tensor.")
    if out.stride(1) != 1:
        raise ValueError("Triton normalization requires contiguous output rows.")


def _block_size(cols: int) -> int:
    if cols <= 0:
        raise ValueError("normalization requires at least one column")
    block_size = 1 << (cols - 1).bit_length()
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(
            "Triton normalization supports up to "
            f"{MAX_BLOCK_SIZE} columns after padding; got {cols}"
        )
    return block_size


def _num_warps(block_size: int) -> int:
    if block_size >= 32_768:
        return 32
    if block_size >= 8_192:
        return 16
    if block_size >= 2_048:
        return 8
    return 4


if triton is not None and tl is not None:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        x_row_stride: tl.constexpr,
        out_row_stride: tl.constexpr,
        cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols

        values = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
        values = values.to(tl.float32)
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / cols
        normalized = values * tl.rsqrt(mean_square + eps)
        output = normalized * weights
        tl.store(out_ptr + row * out_row_stride + offsets, output, mask=mask)

    @triton.jit
    def _layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        x_row_stride: tl.constexpr,
        out_row_stride: tl.constexpr,
        cols: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols

        values = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0)
        values = values.to(tl.float32)
        mean = tl.sum(values, axis=0) / cols
        centered = values - mean
        variance = tl.sum(centered * centered, axis=0) / cols
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = centered * tl.rsqrt(variance + eps) * weights + bias
        tl.store(out_ptr + row * out_row_stride + offsets, output, mask=mask)

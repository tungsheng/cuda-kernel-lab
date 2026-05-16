"""Triton fused row-wise softmax."""

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

    return bool(
        torch is not None
        and triton is not None
        and torch.cuda.is_available()
    )


def softmax(x: Any) -> Any:
    """Return row-wise softmax over the last dimension using one fused kernel."""

    _require_triton_matrix(x)
    rows, cols = x.shape
    block_size = _next_power_of_2(cols)
    if block_size > MAX_BLOCK_SIZE:
        raise ValueError(
            f"Triton softmax supports up to {MAX_BLOCK_SIZE} columns after padding; got {cols}"
        )

    out = torch.empty_like(x)
    _softmax_kernel[(rows,)](
        x,
        out,
        x.stride(0),
        out.stride(0),
        cols,
        block_size,
        num_warps=_num_warps(block_size),
    )
    return out


def _require_triton_matrix(x: Any) -> None:
    if torch is None or triton is None:
        raise RuntimeError("Triton softmax requires torch and triton to be installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton softmax requires a CUDA-capable PyTorch runtime.")
    if not getattr(x, "is_cuda", False):
        raise RuntimeError("Triton softmax requires CUDA tensors.")
    if getattr(x, "ndim", None) != 2:
        ndim = getattr(x, "ndim", None)
        raise ValueError(f"row-wise softmax requires a 2D tensor, got ndim={ndim}")
    if x.shape[1] <= 0:
        raise ValueError("row-wise softmax requires at least one column")
    if x.stride(1) != 1:
        raise ValueError("Triton softmax requires the last dimension to be contiguous.")


def _next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


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
    def _softmax_kernel(
        x_ptr,
        out_ptr,
        x_row_stride: tl.constexpr,
        out_row_stride: tl.constexpr,
        cols: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < cols

        values = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        shifted = values - tl.max(values, axis=0)
        numerator = tl.exp(shifted)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator

        tl.store(out_ptr + row * out_row_stride + offsets, output, mask=mask)

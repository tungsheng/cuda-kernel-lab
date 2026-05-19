"""Triton implementations for memory-bandwidth primitives."""

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


DEFAULT_BLOCK_SIZE = 1024


def is_available() -> bool:
    """Return true when Triton and a CUDA-capable PyTorch runtime are available."""

    return bool(torch is not None and triton is not None and torch.cuda.is_available())


def vector_add(
    a: Any,
    b: Any,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    out: Any | None = None,
) -> Any:
    """Return elementwise a + b using a Triton kernel."""

    _require_same_shape(a, b)
    _require_triton_tensor(a)
    _require_triton_tensor(b)
    out = torch.empty_like(a) if out is None else out
    _require_same_shape(a, out)
    _require_triton_tensor(out)
    _vector_add_kernel[_grid(a.numel(), block_size)](a, b, out, a.numel(), block_size)
    return out


def copy(x: Any, *, block_size: int = DEFAULT_BLOCK_SIZE, out: Any | None = None) -> Any:
    """Return a materialized copy of x using a Triton kernel."""

    _require_triton_tensor(x)
    out = torch.empty_like(x) if out is None else out
    _require_same_shape(x, out)
    _require_triton_tensor(out)
    _copy_kernel[_grid(x.numel(), block_size)](x, out, x.numel(), block_size)
    return out


def scale(
    x: Any,
    alpha: float,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    out: Any | None = None,
) -> Any:
    """Return x scaled by alpha using a Triton kernel."""

    _require_triton_tensor(x)
    out = torch.empty_like(x) if out is None else out
    _require_same_shape(x, out)
    _require_triton_tensor(out)
    _scale_kernel[_grid(x.numel(), block_size)](x, out, alpha, x.numel(), block_size)
    return out


def reduction_sum(
    x: Any,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    strategy: str = "iterative",
) -> Any:
    """Return the sum of x using a selected Triton reduction strategy.

    Partial sums accumulate in FP32 so low-precision inputs avoid the worst
    reduction-order error. The returned scalar is FP32.
    """

    _require_triton_tensor(x)
    if strategy == "two_pass":
        return _reduction_sum_two_pass(x, block_size=block_size)
    if strategy != "iterative":
        raise ValueError(f"unknown reduction strategy: {strategy}")

    return _reduction_sum_iterative(x, block_size=block_size)


def _reduction_sum_iterative(x: Any, *, block_size: int) -> Any:
    current = x
    current_numel = x.numel()

    while current_numel > 1:
        blocks = triton.cdiv(current_numel, block_size)
        partials = torch.empty((blocks,), device=x.device, dtype=torch.float32)
        _reduction_sum_kernel[(blocks,)](current, partials, current_numel, block_size)
        current = partials
        current_numel = blocks

    if current_numel == 0:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    return current.reshape(())


def _reduction_sum_two_pass(x: Any, *, block_size: int) -> Any:
    current_numel = x.numel()
    if current_numel == 0:
        return torch.zeros((), device=x.device, dtype=torch.float32)

    blocks = triton.cdiv(current_numel, block_size)
    partials = torch.empty((blocks,), device=x.device, dtype=torch.float32)
    _reduction_sum_kernel[(blocks,)](x, partials, current_numel, block_size)
    return partials.sum()


def _grid(numel: int, block_size: int) -> tuple[int]:
    return (triton.cdiv(numel, block_size),)


def _require_triton_tensor(x: Any) -> None:
    if torch is None or triton is None:
        raise RuntimeError("Triton memory kernels require torch and triton to be installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton memory kernels require a CUDA-capable PyTorch runtime.")
    if not getattr(x, "is_cuda", False):
        raise RuntimeError("Triton memory kernels require CUDA tensors.")
    if x.numel() < 0:
        raise ValueError("tensor numel must be non-negative")


def _require_same_shape(a: Any, b: Any) -> None:
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        left = getattr(a, "shape", None)
        right = getattr(b, "shape", None)
        raise ValueError(f"shape mismatch: {left} != {right}")


if triton is not None and tl is not None:

    @triton.jit
    def _vector_add_kernel(a_ptr, b_ptr, out_ptr, numel: tl.constexpr, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)

    @triton.jit
    def _copy_kernel(x_ptr, out_ptr, numel: tl.constexpr, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, values, mask=mask)

    @triton.jit
    def _scale_kernel(x_ptr, out_ptr, alpha, numel: tl.constexpr, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, values * alpha, mask=mask)

    @triton.jit
    def _reduction_sum_kernel(x_ptr, partials_ptr, numel: tl.constexpr, block_size: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partial = tl.sum(values, axis=0)
        tl.store(partials_ptr + pid, partial)

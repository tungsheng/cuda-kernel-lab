"""Triton tiled matrix multiplication."""

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


DEFAULT_BLOCK_M = 16
DEFAULT_BLOCK_N = 16
DEFAULT_BLOCK_K = 32
DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 3
DEFAULT_INPUT_PRECISION = "ieee"
INPUT_PRECISIONS = ("tf32", "tf32x3", "ieee")


def is_available() -> bool:
    """Return true when Triton and a CUDA-capable PyTorch runtime are available."""

    return bool(torch is not None and triton is not None and torch.cuda.is_available())


def matmul(
    a: Any,
    b: Any,
    *,
    block_m: int = DEFAULT_BLOCK_M,
    block_n: int = DEFAULT_BLOCK_N,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = DEFAULT_NUM_WARPS,
    num_stages: int = DEFAULT_NUM_STAGES,
    input_precision: str = DEFAULT_INPUT_PRECISION,
    out: Any | None = None,
) -> Any:
    """Return a @ b using a tiled Triton kernel."""

    _require_positive_blocks(block_m=block_m, block_n=block_n, block_k=block_k)
    _require_positive_launch(num_warps=num_warps, num_stages=num_stages)
    _require_input_precision(input_precision)
    _require_matmul_inputs(a, b)
    m, k = a.shape
    _, n = b.shape
    out = torch.empty((m, n), device=a.device, dtype=a.dtype) if out is None else out
    _require_matmul_output(out, m=m, n=n)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        block_m,
        block_n,
        block_k,
        input_precision,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _require_positive_blocks(*, block_m: int, block_n: int, block_k: int) -> None:
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")


def _require_positive_launch(*, num_warps: int, num_stages: int) -> None:
    if num_warps <= 0 or num_stages <= 0:
        raise ValueError("num_warps and num_stages must be positive")


def _require_input_precision(input_precision: str) -> None:
    if input_precision not in INPUT_PRECISIONS:
        choices = ", ".join(INPUT_PRECISIONS)
        raise ValueError(f"input_precision must be one of: {choices}")


def _require_matmul_inputs(a: Any, b: Any) -> None:
    if torch is None or triton is None:
        raise RuntimeError("Triton matmul requires torch and triton to be installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton matmul requires a CUDA-capable PyTorch runtime.")
    if not getattr(a, "is_cuda", False) or not getattr(b, "is_cuda", False):
        raise RuntimeError("Triton matmul inputs must be CUDA tensors.")
    if getattr(a, "ndim", None) != 2 or getattr(b, "ndim", None) != 2:
        left = getattr(a, "ndim", None)
        right = getattr(b, "ndim", None)
        raise ValueError(f"matmul requires two 2D tensors, got ndim={left} and ndim={right}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {a.shape} cannot multiply {b.shape}")
    if a.device != b.device:
        raise ValueError("Triton matmul inputs must be on the same device.")
    if a.dtype != b.dtype:
        raise ValueError("Triton matmul inputs must have the same dtype.")


def _require_matmul_output(out: Any, *, m: int, n: int) -> None:
    if getattr(out, "shape", None) != (m, n):
        raise ValueError(f"output shape must be ({m}, {n}), got {getattr(out, 'shape', None)}")
    if not getattr(out, "is_cuda", False):
        raise RuntimeError("Triton matmul output must be a CUDA tensor.")


if triton is not None and tl is not None:

    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        m: tl.constexpr,
        n: tl.constexpr,
        k: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        input_precision: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(n, block_n)
        pid_m = pid // num_pid_n
        pid_n = pid - pid_m * num_pid_n

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        accumulator = tl.zeros((block_m, block_n), tl.float32)

        for k_start in range(0, k, block_k):
            k_offsets = k_start + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + k_offsets[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (k_offsets[None, :] < k),
                other=0.0,
            )
            b = tl.load(
                b_ptr + k_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(k_offsets[:, None] < k) & (offs_n[None, :] < n),
                other=0.0,
            )
            accumulator += tl.dot(a, b, input_precision=input_precision)

        tl.store(
            out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            accumulator,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

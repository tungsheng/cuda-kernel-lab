"""Triton fused SwiGLU elementwise kernel."""

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


def swiglu(
    gate: Any,
    up: Any,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    out: Any | None = None,
) -> Any:
    """Return SiLU(gate) * up using one fused Triton kernel."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    _require_swiglu_input(gate, "gate")
    _require_swiglu_input(up, "up")
    _require_same_shape(gate, up)
    out = torch.empty_like(gate) if out is None else out
    _require_swiglu_output(gate, out)
    _swiglu_kernel[_grid(gate.numel(), block_size)](
        gate,
        up,
        out,
        gate.numel(),
        block_size,
    )
    return out


def _grid(numel: int, block_size: int) -> tuple[int]:
    return (triton.cdiv(numel, block_size),)


def _require_swiglu_input(x: Any, name: str) -> None:
    if torch is None or triton is None:
        raise RuntimeError("Triton SwiGLU requires torch and triton to be installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("Triton SwiGLU requires a CUDA-capable PyTorch runtime.")
    if not getattr(x, "is_cuda", False):
        raise RuntimeError(f"Triton SwiGLU {name} tensor must be a CUDA tensor.")
    if not x.is_contiguous():
        raise ValueError(f"Triton SwiGLU {name} tensor must be contiguous.")


def _require_swiglu_output(gate: Any, out: Any) -> None:
    _require_same_shape(gate, out)
    if not getattr(out, "is_cuda", False):
        raise RuntimeError("Triton SwiGLU output must be a CUDA tensor.")
    if not out.is_contiguous():
        raise ValueError("Triton SwiGLU output must be contiguous.")


def _require_same_shape(a: Any, b: Any) -> None:
    if getattr(a, "shape", None) != getattr(b, "shape", None):
        left = getattr(a, "shape", None)
        right = getattr(b, "shape", None)
        raise ValueError(f"shape mismatch: {left} != {right}")


if triton is not None and tl is not None:

    @triton.jit
    def _swiglu_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        numel: tl.constexpr,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < numel

        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        output = gate * sigmoid * up
        tl.store(out_ptr + offsets, output, mask=mask)

"""Benchmark SwiGLU elementwise fusion across available backends."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from cuda_kernel_lab.benchmark import BenchmarkResult, benchmark_callable, check_tensors_close
from cuda_kernel_lab.benchmark_cli import (
    add_common_benchmark_args,
    correctness_tolerance,
    dtype_label,
    emit_results,
    ensure_backend_available,
    print_results_table,
    require_torch,
    resolve_device,
    resolve_dtype,
    selected_backends,
)
from cuda_kernel_lab.metrics import dtype_size_bytes
from cuda_kernel_lab.ops.swiglu import flop_count, memory_traffic_bytes


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    results = []
    for backend in selected_backends(args.backend, device, triton_available=triton_is_available):
        ensure_backend_available(backend, device, triton_available=triton_is_available)
        results.append(
            run_one(
                torch=torch,
                backend=backend,
                rows=args.rows,
                cols=args.cols,
                dtype=dtype,
                device=device,
                block_size=args.block_size,
                warmup=args.warmup,
                iterations=args.iterations,
                skip_correctness=args.skip_correctness,
            )
        )

    emit_results(results, args=args, benchmark="swiglu", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Triton block size for the fused SwiGLU kernel; ignored by torch.",
    )
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton.swiglu import is_available
    except ImportError:
        return False
    return is_available()


def run_one(
    *,
    torch: Any,
    backend: str,
    rows: int,
    cols: int,
    dtype: Any,
    device: str,
    block_size: int,
    warmup: int,
    iterations: int,
    skip_correctness: bool,
) -> BenchmarkResult:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    gate = torch.randn((rows, cols), device=device, dtype=dtype)
    up = torch.randn((rows, cols), device=device, dtype=dtype)
    out = torch.empty_like(gate)
    fn = build_op(backend, gate, up, out, block_size)
    correctness = None
    if not skip_correctness:
        expected = build_op("torch", gate, up, None, block_size)()
        actual = fn()
        rtol, atol = correctness_tolerance(dtype)
        correctness = check_tensors_close(
            actual,
            expected,
            torch=torch,
            rtol=rtol,
            atol=atol,
        )

    numel = rows * cols
    dtype_size = dtype_size_bytes(dtype)

    return benchmark_callable(
        f"{backend}:swiglu",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(rows, cols),
        bytes_moved=memory_traffic_bytes(numel=numel, dtype_size=dtype_size),
        flops=flop_count(numel=numel),
        warmup=warmup,
        iterations=iterations,
        strategy="torch-baseline" if backend == "torch" else "triton-fused-swiglu",
        variant=f"block_size={block_size}",
        parameters={"block_size": block_size},
        correctness=correctness,
    )


def build_op(
    backend: str,
    gate: Any,
    up: Any,
    out: Any | None,
    block_size: int,
) -> Callable[[], Any]:
    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import swiglu

        return lambda: swiglu(gate, up, out=out)

    if backend == "triton":
        from cuda_kernel_lab.kernels.triton import swiglu

        return lambda: swiglu(gate, up, block_size=block_size, out=out)

    raise ValueError(f"unknown backend: {backend}")


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=16, include_shape=True)


if __name__ == "__main__":
    main()

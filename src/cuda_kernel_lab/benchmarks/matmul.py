"""Benchmark matrix multiplication across available backends."""

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
from cuda_kernel_lab.ops.matmul import flop_count, memory_traffic_bytes
from cuda_kernel_lab.optimization import matmul_optimization


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
                m=args.m,
                n=args.n,
                k=args.k,
                dtype=dtype,
                device=device,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                warmup=args.warmup,
                iterations=args.iterations,
                skip_correctness=args.skip_correctness,
            )
        )

    emit_results(results, args=args, benchmark="matmul", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    parser.add_argument("--block-k", type=int, default=32)
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton.matmul import is_available
    except ImportError:
        return False
    return is_available()


def run_one(
    *,
    torch: Any,
    backend: str,
    m: int,
    n: int,
    k: int,
    dtype: Any,
    device: str,
    block_m: int,
    block_n: int,
    block_k: int,
    warmup: int,
    iterations: int,
    skip_correctness: bool,
) -> BenchmarkResult:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    out = torch.empty((m, n), device=device, dtype=dtype)
    fn = build_op(backend, a, b, out, block_m, block_n, block_k)
    correctness = None
    if not skip_correctness:
        expected = build_op("torch", a, b, None, block_m, block_n, block_k)()
        actual = fn()
        rtol, atol = correctness_tolerance(dtype)
        correctness = check_tensors_close(
            actual,
            expected,
            torch=torch,
            rtol=rtol,
            atol=atol,
        )

    dtype_size = dtype_size_bytes(dtype)

    return benchmark_callable(
        f"{backend}:matmul",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(m, n, k),
        bytes_moved=memory_traffic_bytes(m=m, n=n, k=k, dtype_size=dtype_size),
        flops=flop_count(m=m, n=n, k=k),
        warmup=warmup,
        iterations=iterations,
        strategy="torch-baseline" if backend == "torch" else "triton-tiled-dot",
        variant=f"block_m={block_m}, block_n={block_n}, block_k={block_k}",
        parameters={
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
        },
        optimization=matmul_optimization(backend=backend),
        correctness=correctness,
    )


def build_op(
    backend: str,
    a: Any,
    b: Any,
    out: Any | None,
    block_m: int,
    block_n: int,
    block_k: int,
) -> Callable[[], Any]:
    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import matmul

        return lambda: matmul(a, b, out=out)

    if backend == "triton":
        from cuda_kernel_lab.kernels.triton import matmul

        return lambda: matmul(
            a,
            b,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            out=out,
        )

    raise ValueError(f"unknown backend: {backend}")


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=16, include_shape=True)


if __name__ == "__main__":
    main()

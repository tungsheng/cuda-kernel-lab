"""Benchmark row-wise softmax across available backends."""

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
from cuda_kernel_lab.ops.softmax import flop_count, memory_traffic_bytes
from cuda_kernel_lab.optimization import softmax_optimization

TRAFFIC_MODELS = ("fused", "naive")


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
                traffic_model=args.traffic_model,
                warmup=args.warmup,
                iterations=args.iterations,
                skip_correctness=args.skip_correctness,
            )
        )

    emit_results(results, args=args, benchmark="softmax", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--traffic-model", choices=TRAFFIC_MODELS, default="fused")
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton.softmax import is_available
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
    traffic_model: str,
    warmup: int,
    iterations: int,
    skip_correctness: bool,
) -> BenchmarkResult:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    x = torch.randn((rows, cols), device=device, dtype=dtype)
    out = torch.empty_like(x) if backend == "triton" else None
    fn = build_op(backend, x, out)
    correctness = None
    if not skip_correctness:
        expected = build_op("torch", x, None)()
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
        f"{backend}:softmax",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(rows, cols),
        bytes_moved=memory_traffic_bytes(
            rows=rows,
            cols=cols,
            dtype_size=dtype_size,
            model=traffic_model,
        ),
        flops=flop_count(rows=rows, cols=cols),
        warmup=warmup,
        iterations=iterations,
        strategy="torch-baseline" if backend == "torch" else "triton-fused-row-softmax",
        variant=f"traffic_model={traffic_model}",
        parameters={"traffic_model": traffic_model},
        optimization=softmax_optimization(backend=backend),
        correctness=correctness,
    )


def build_op(backend: str, x: Any, out: Any | None) -> Callable[[], Any]:
    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import softmax

        return lambda: softmax(x)

    if backend == "triton":
        from cuda_kernel_lab.kernels.triton import softmax

        return lambda: softmax(x, out=out)

    raise ValueError(f"unknown backend: {backend}")


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=16, include_shape=True)


if __name__ == "__main__":
    main()

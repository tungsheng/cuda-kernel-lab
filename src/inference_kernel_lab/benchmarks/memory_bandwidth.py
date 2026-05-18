"""Benchmark memory-bandwidth primitives across available backends."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from inference_kernel_lab.benchmark import BenchmarkResult, benchmark_callable
from inference_kernel_lab.benchmark_cli import (
    add_common_benchmark_args,
    dtype_label,
    emit_results,
    ensure_backend_available,
    print_results_table,
    require_torch,
    resolve_device,
    resolve_dtype,
    selected_backends,
    selected_ops,
)
from inference_kernel_lab.metrics import dtype_size_bytes
from inference_kernel_lab.ops.memory import flop_count, memory_traffic_bytes

OPS = ("copy", "scale", "vector_add", "reduction_sum")


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    backends = selected_backends(args.backend, device, triton_available=triton_is_available)
    results = []
    for backend in backends:
        ensure_backend_available(backend, device, triton_available=triton_is_available)
        for op_name in selected_ops(args.op, OPS):
            results.append(
                run_one(
                    torch=torch,
                    backend=backend,
                    op_name=op_name,
                    numel=args.numel,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )

    emit_results(
        results,
        args=args,
        benchmark="memory_bandwidth",
        print_table=print_table,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", *OPS), default="all")
    parser.add_argument("--numel", type=int, default=16_777_216)
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from inference_kernel_lab.kernels.triton import memory
    except ImportError:
        return False
    return memory.is_available()


def run_one(
    *,
    torch: Any,
    backend: str,
    op_name: str,
    numel: int,
    dtype: Any,
    device: str,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    if numel <= 0:
        raise ValueError("numel must be positive")

    x = torch.randn(numel, device=device, dtype=dtype)
    y = torch.randn(numel, device=device, dtype=dtype)
    out = None if op_name == "reduction_sum" else torch.empty_like(x)
    fn = build_op(backend, op_name, x, y, out)
    dtype_size = dtype_size_bytes(dtype)

    return benchmark_callable(
        f"{backend}:{op_name}",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(numel,),
        bytes_moved=memory_traffic_bytes(op_name, numel=numel, dtype_size=dtype_size),
        flops=flop_count(op_name, numel=numel),
        warmup=warmup,
        iterations=iterations,
    )


def build_op(
    backend: str,
    op_name: str,
    x: Any,
    y: Any,
    out: Any | None,
) -> Callable[[], Any]:
    if backend == "torch":
        from inference_kernel_lab.kernels.torch_baselines import memory

        if op_name == "copy":
            return lambda: memory.copy(x, out=out)
        if op_name == "scale":
            return lambda: memory.scale(x, 0.5, out=out)
        if op_name == "vector_add":
            return lambda: memory.vector_add(x, y, out=out)
        if op_name == "reduction_sum":
            return lambda: memory.reduction_sum(x)

    if backend == "triton":
        from inference_kernel_lab.kernels.triton import memory

        if op_name == "copy":
            return lambda: memory.copy(x, out=out)
        if op_name == "scale":
            return lambda: memory.scale(x, 0.5, out=out)
        if op_name == "vector_add":
            return lambda: memory.vector_add(x, y, out=out)
        if op_name == "reduction_sum":
            return lambda: memory.reduction_sum(x)

    raise ValueError(f"unknown backend/op combination: {backend}:{op_name}")


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=22, include_shape=False)


if __name__ == "__main__":
    main()

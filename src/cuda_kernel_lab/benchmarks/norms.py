"""Benchmark row-wise RMSNorm and LayerNorm across available backends."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from cuda_kernel_lab.benchmark import BenchmarkResult, benchmark_callable
from cuda_kernel_lab.benchmark_cli import (
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
from cuda_kernel_lab.metrics import dtype_size_bytes
from cuda_kernel_lab.ops.norms import flop_count, memory_traffic_bytes

OPS = ("rmsnorm", "layernorm")


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    results = []
    for backend in selected_backends(args.backend, device, triton_available=triton_is_available):
        ensure_backend_available(backend, device, triton_available=triton_is_available)
        for op_name in selected_ops(args.op, OPS):
            results.append(
                run_one(
                    torch=torch,
                    backend=backend,
                    op_name=op_name,
                    rows=args.rows,
                    cols=args.cols,
                    dtype=dtype,
                    device=device,
                    eps=args.eps,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )

    emit_results(results, args=args, benchmark="norms", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", *OPS), default="all")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--eps", type=float, default=None)
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton.norms import is_available
    except ImportError:
        return False
    return is_available()


def run_one(
    *,
    torch: Any,
    backend: str,
    op_name: str,
    rows: int,
    cols: int,
    dtype: Any,
    device: str,
    eps: float | None,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    x = torch.randn((rows, cols), device=device, dtype=dtype)
    weight = torch.randn((cols,), device=device, dtype=dtype)
    bias = torch.randn((cols,), device=device, dtype=dtype)
    out = torch.empty_like(x) if backend == "triton" else None
    fn = build_op(backend, op_name, x, weight, bias, eps, out)
    dtype_size = dtype_size_bytes(dtype)

    return benchmark_callable(
        f"{backend}:{op_name}",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(rows, cols),
        bytes_moved=memory_traffic_bytes(
            op_name,
            rows=rows,
            cols=cols,
            dtype_size=dtype_size,
        ),
        flops=flop_count(op_name, rows=rows, cols=cols),
        warmup=warmup,
        iterations=iterations,
    )


def build_op(
    backend: str,
    op_name: str,
    x: Any,
    weight: Any,
    bias: Any,
    eps: float | None,
    out: Any | None,
) -> Callable[[], Any]:
    effective_eps = default_eps(op_name) if eps is None else eps

    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import layernorm, rmsnorm

        if op_name == "rmsnorm":
            return lambda: rmsnorm(x, weight, eps=effective_eps)
        if op_name == "layernorm":
            return lambda: layernorm(x, weight, bias, eps=effective_eps)

    if backend == "triton":
        from cuda_kernel_lab.kernels.triton import layernorm, rmsnorm

        if op_name == "rmsnorm":
            return lambda: rmsnorm(x, weight, eps=effective_eps, out=out)
        if op_name == "layernorm":
            return lambda: layernorm(x, weight, bias, eps=effective_eps, out=out)

    raise ValueError(f"unknown backend/op combination: {backend}:{op_name}")


def default_eps(op_name: str) -> float:
    if op_name == "rmsnorm":
        return 1e-6
    if op_name == "layernorm":
        return 1e-5
    raise ValueError(f"unknown normalization primitive: {op_name}")


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=18, include_shape=True)


if __name__ == "__main__":
    main()

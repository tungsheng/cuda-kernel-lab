"""Benchmark memory-bandwidth primitives across available backends."""

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
    selected_ops,
)
from cuda_kernel_lab.metrics import dtype_size_bytes
from cuda_kernel_lab.ops.memory import flop_count, memory_traffic_bytes, reduction_traffic_bytes
from cuda_kernel_lab.optimization import memory_optimization

OPS = ("copy", "scale", "vector_add", "reduction_sum")
REDUCTION_STRATEGIES = ("iterative", "two_pass")


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
                    block_size=args.block_size,
                    reduction_strategy=args.reduction_strategy,
                    skip_correctness=args.skip_correctness,
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Triton block size for memory kernels; ignored by the torch backend.",
    )
    parser.add_argument(
        "--reduction-strategy",
        choices=REDUCTION_STRATEGIES,
        default="iterative",
        help="Reduction implementation strategy for reduction_sum.",
    )
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton import memory
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
    block_size: int,
    reduction_strategy: str,
    skip_correctness: bool,
) -> BenchmarkResult:
    if numel <= 0:
        raise ValueError("numel must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    x = torch.randn(numel, device=device, dtype=dtype)
    y = torch.randn(numel, device=device, dtype=dtype)
    out = None if op_name == "reduction_sum" else torch.empty_like(x)
    fn = build_op(backend, op_name, x, y, out, block_size, reduction_strategy)
    correctness = None
    if not skip_correctness:
        expected = build_reference_op(op_name, x, y)()
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
    bytes_moved = (
        reduction_traffic_bytes(
            numel=numel,
            dtype_size=dtype_size,
            block_size=block_size,
            strategy=reduction_strategy,
        )
        if op_name == "reduction_sum"
        else memory_traffic_bytes(op_name, numel=numel, dtype_size=dtype_size)
    )

    return benchmark_callable(
        f"{backend}:{op_name}",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(numel,),
        bytes_moved=bytes_moved,
        flops=flop_count(op_name, numel=numel),
        warmup=warmup,
        iterations=iterations,
        strategy=_strategy_label(backend, op_name, reduction_strategy),
        variant=_variant_label(op_name, block_size, reduction_strategy),
        parameters={
            "block_size": block_size,
            "reduction_strategy": reduction_strategy,
        },
        optimization=memory_optimization(
            backend=backend,
            op_name=op_name,
            reduction_strategy=reduction_strategy,
        ),
        correctness=correctness,
    )


def build_op(
    backend: str,
    op_name: str,
    x: Any,
    y: Any,
    out: Any | None,
    block_size: int,
    reduction_strategy: str,
) -> Callable[[], Any]:
    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import memory

        if op_name == "copy":
            return lambda: memory.copy(x, out=out)
        if op_name == "scale":
            return lambda: memory.scale(x, 0.5, out=out)
        if op_name == "vector_add":
            return lambda: memory.vector_add(x, y, out=out)
        if op_name == "reduction_sum":
            return lambda: memory.reduction_sum(x)

    if backend == "triton":
        from cuda_kernel_lab.kernels.triton import memory

        if op_name == "copy":
            return lambda: memory.copy(x, block_size=block_size, out=out)
        if op_name == "scale":
            return lambda: memory.scale(x, 0.5, block_size=block_size, out=out)
        if op_name == "vector_add":
            return lambda: memory.vector_add(x, y, block_size=block_size, out=out)
        if op_name == "reduction_sum":
            return lambda: memory.reduction_sum(
                x,
                block_size=block_size,
                strategy=reduction_strategy,
            )

    raise ValueError(f"unknown backend/op combination: {backend}:{op_name}")


def build_reference_op(op_name: str, x: Any, y: Any) -> Callable[[], Any]:
    from cuda_kernel_lab.kernels.torch_baselines import memory

    if op_name == "copy":
        return lambda: memory.copy(x)
    if op_name == "scale":
        return lambda: memory.scale(x, 0.5)
    if op_name == "vector_add":
        return lambda: memory.vector_add(x, y)
    if op_name == "reduction_sum":
        return lambda: memory.reduction_sum(x)
    raise ValueError(f"unknown memory primitive: {op_name}")


def _strategy_label(backend: str, op_name: str, reduction_strategy: str) -> str:
    if backend == "torch":
        return "torch-baseline"
    if op_name == "reduction_sum":
        return f"triton-reduction-{reduction_strategy.replace('_', '-')}"
    return "triton-block-size"


def _variant_label(op_name: str, block_size: int, reduction_strategy: str) -> str:
    if op_name == "reduction_sum":
        return f"reduction_strategy={reduction_strategy}, block_size={block_size}"
    return f"block_size={block_size}"


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=22, include_shape=False)


if __name__ == "__main__":
    main()

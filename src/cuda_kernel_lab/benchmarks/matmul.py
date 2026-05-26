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

INPUT_PRECISIONS = ("tf32", "tf32x3", "ieee")
SCHEDULES = ("standard", "persistent")
DEFAULT_SCHEDULE = "standard"


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
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                input_precision=args.input_precision,
                group_m=args.group_m,
                schedule=args.schedule,
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
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument(
        "--group-m",
        type=int,
        default=1,
        help="Number of M tiles grouped together when mapping Triton programs.",
    )
    parser.add_argument(
        "--input-precision",
        choices=INPUT_PRECISIONS,
        default="ieee",
        help="Triton tl.dot input precision for float32 inputs.",
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default=DEFAULT_SCHEDULE,
        help="Triton matmul tile scheduler.",
    )
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
    num_warps: int,
    num_stages: int,
    input_precision: str,
    group_m: int,
    schedule: str,
    warmup: int,
    iterations: int,
    skip_correctness: bool,
) -> BenchmarkResult:
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError("m, n, and k must be positive")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")
    if num_warps <= 0 or num_stages <= 0 or group_m <= 0:
        raise ValueError("num_warps, num_stages, and group_m must be positive")
    if input_precision not in INPUT_PRECISIONS:
        choices = ", ".join(INPUT_PRECISIONS)
        raise ValueError(f"input_precision must be one of: {choices}")
    if schedule not in SCHEDULES:
        choices = ", ".join(SCHEDULES)
        raise ValueError(f"schedule must be one of: {choices}")

    a = torch.randn((m, k), device=device, dtype=dtype)
    b = torch.randn((k, n), device=device, dtype=dtype)
    out = torch.empty((m, n), device=device, dtype=dtype)
    fn = build_op(
        backend,
        a,
        b,
        out,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        input_precision,
        group_m,
        schedule,
    )
    correctness = None
    if not skip_correctness:
        expected = build_op(
            "torch",
            a,
            b,
            None,
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            input_precision,
            group_m,
            DEFAULT_SCHEDULE,
        )()
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

    strategy = _strategy_label(backend=backend, schedule=schedule)
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
        strategy=strategy,
        variant=(
            f"block_m={block_m}, block_n={block_n}, block_k={block_k}, "
            f"num_warps={num_warps}, num_stages={num_stages}, "
            f"input_precision={input_precision}, group_m={group_m}, schedule={schedule}"
        ),
        parameters={
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "input_precision": input_precision,
            "group_m": group_m,
            "schedule": schedule,
        },
        optimization=matmul_optimization(backend=backend, schedule=schedule),
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
    num_warps: int,
    num_stages: int,
    input_precision: str,
    group_m: int,
    schedule: str,
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
            num_warps=num_warps,
            num_stages=num_stages,
            input_precision=input_precision,
            group_m=group_m,
            schedule=schedule,
            out=out,
        )

    raise ValueError(f"unknown backend: {backend}")


def _strategy_label(*, backend: str, schedule: str) -> str:
    if backend == "torch":
        return "torch-baseline"
    if schedule == "persistent":
        return "triton-persistent-tiled-dot"
    return "triton-tiled-dot"


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=16, include_shape=True)


if __name__ == "__main__":
    main()

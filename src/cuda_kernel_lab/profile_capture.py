"""Small CUDA profiler capture harnesses for Nsight Compute runs."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from cuda_kernel_lab.benchmark_cli import require_torch, resolve_device, resolve_dtype
from cuda_kernel_lab.benchmarks.matmul import (
    DEFAULT_SCHEDULE,
    INPUT_PRECISIONS,
    SCHEDULES,
    build_op,
)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "matmul":
        run_matmul(args)
        return
    raise ValueError(f"unknown profile capture command: {args.command}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    matmul = subparsers.add_parser("matmul", help="Capture one Triton matmul kernel.")
    matmul.add_argument("--m", type=int, required=True)
    matmul.add_argument("--n", type=int, required=True)
    matmul.add_argument("--k", type=int, required=True)
    matmul.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    matmul.add_argument("--device", default="cuda", choices=("cuda",))
    matmul.add_argument("--block-m", type=int, required=True)
    matmul.add_argument("--block-n", type=int, required=True)
    matmul.add_argument("--block-k", type=int, required=True)
    matmul.add_argument("--num-warps", type=int, required=True)
    matmul.add_argument("--num-stages", type=int, required=True)
    matmul.add_argument("--input-precision", choices=INPUT_PRECISIONS, required=True)
    matmul.add_argument("--group-m", type=int, required=True)
    matmul.add_argument("--schedule", choices=SCHEDULES, default=DEFAULT_SCHEDULE)
    matmul.add_argument("--warmup", type=int, default=2)
    matmul.add_argument("--profile-iterations", type=int, default=1)

    return parser.parse_args(argv)


def run_matmul(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.profile_iterations <= 0:
        raise ValueError("profile-iterations must be positive")

    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    a = torch.randn((args.m, args.k), device=device, dtype=dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=dtype)
    out = torch.empty((args.m, args.n), device=device, dtype=dtype)
    fn = build_op(
        "triton",
        a,
        b,
        out,
        args.block_m,
        args.block_n,
        args.block_k,
        args.num_warps,
        args.num_stages,
        args.input_precision,
        args.group_m,
        args.schedule,
    )

    _run_warmup(fn, torch=torch, device=device, warmup=args.warmup)
    _profile(fn, torch=torch, device=device, iterations=args.profile_iterations)


def _run_warmup(
    fn: Callable[[], Any],
    *,
    torch: Any,
    device: str,
    warmup: int,
) -> None:
    for _ in range(warmup):
        fn()
    _synchronize(torch, device)


def _profile(
    fn: Callable[[], Any],
    *,
    torch: Any,
    device: str,
    iterations: int,
) -> None:
    _cuda_profiler_start(torch)
    try:
        for _ in range(iterations):
            fn()
        _synchronize(torch, device)
    finally:
        _cuda_profiler_stop(torch)


def _cuda_profiler_start(torch: Any) -> None:
    if hasattr(torch.cuda, "cudart"):
        torch.cuda.cudart().cudaProfilerStart()
        return
    torch.cuda.profiler.start()


def _cuda_profiler_stop(torch: Any) -> None:
    if hasattr(torch.cuda, "cudart"):
        torch.cuda.cudart().cudaProfilerStop()
        return
    torch.cuda.profiler.stop()


def _synchronize(torch: Any, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()

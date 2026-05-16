"""Benchmark PyTorch baselines for memory-bandwidth primitives."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from inference_kernel_lab.benchmark import BenchmarkResult, benchmark_callable
from inference_kernel_lab.metrics import dtype_size_bytes
from kernels.torch_baselines.memory import flop_count, memory_traffic_bytes

OPS = ("copy", "scale", "vector_add", "reduction_sum")


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    results = [
        run_one(
            torch=torch,
            op_name=op_name,
            numel=args.numel,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for op_name in selected_ops(args.op)
    ]

    if args.json:
        print(json.dumps([result.as_dict() for result in results], indent=2))
    else:
        print_table(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=("all", *OPS), default="all")
    parser.add_argument("--numel", type=int, default=16_777_216)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def selected_ops(op_name: str) -> tuple[str, ...]:
    return OPS if op_name == "all" else (op_name,)


def require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for benchmarks. Install with: "
            "uv sync --group dev --extra gpu"
        ) from exc
    return torch


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def resolve_dtype(torch: Any, dtype_name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def run_one(
    *,
    torch: Any,
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
    fn = build_op(op_name, x, y)
    dtype_size = dtype_size_bytes(dtype)

    return benchmark_callable(
        op_name,
        fn,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        shape=(numel,),
        bytes_moved=memory_traffic_bytes(op_name, numel=numel, dtype_size=dtype_size),
        flops=flop_count(op_name, numel=numel),
        warmup=warmup,
        iterations=iterations,
    )


def build_op(op_name: str, x: Any, y: Any) -> Callable[[], Any]:
    from kernels.torch_baselines import memory

    if op_name == "copy":
        return lambda: memory.copy(x)
    if op_name == "scale":
        return lambda: memory.scale(x, 0.5)
    if op_name == "vector_add":
        return lambda: memory.vector_add(x, y)
    if op_name == "reduction_sum":
        return lambda: memory.reduction_sum(x)
    raise ValueError(f"unknown op: {op_name}")


def print_table(results: list[BenchmarkResult]) -> None:
    print(
        f"{'name':<15} {'device':<8} {'dtype':<9} "
        f"{'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'GB/s':>10} {'TFLOP/s':>10}"
    )
    for result in results:
        print(
            f"{result.name:<15} {result.device:<8} {result.dtype:<9} "
            f"{result.p50_ms:10.4f} {result.p95_ms:10.4f} {result.p99_ms:10.4f} "
            f"{result.bandwidth_gbps:10.2f} {result.tflops:10.4f}"
        )


if __name__ == "__main__":
    main()

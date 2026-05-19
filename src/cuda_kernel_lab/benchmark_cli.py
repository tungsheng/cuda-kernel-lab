"""Shared command-line helpers for benchmark entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from typing import Any

from cuda_kernel_lab.benchmark import (
    BenchmarkResult,
    benchmark_records,
    collect_run_metadata,
    write_jsonl,
)

BACKENDS = ("torch", "triton")
DTYPES = ("float32", "float16", "bfloat16")
DEVICES = ("auto", "cpu", "cuda")


def add_common_benchmark_args(parser: argparse.ArgumentParser) -> None:
    """Add backend/device/dtype/timing/output arguments shared by all benchmarks."""

    parser.add_argument("--backend", choices=("all", *BACKENDS), default="torch")
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the pre-benchmark PyTorch correctness check.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON records with run metadata.")
    parser.add_argument(
        "--output",
        help="Append JSONL benchmark records to this path, including run metadata.",
    )


def selected_ops(op_name: str, all_ops: Sequence[str]) -> tuple[str, ...]:
    """Resolve an operation selector that may include the sentinel value all."""

    return tuple(all_ops) if op_name == "all" else (op_name,)


def selected_backends(
    backend: str,
    device: str,
    *,
    triton_available: Callable[[], bool],
) -> tuple[str, ...]:
    """Resolve a backend selector that may include the sentinel value all."""

    if backend != "all":
        return (backend,)

    if device == "cuda" and triton_available():
        return BACKENDS
    return ("torch",)


def ensure_backend_available(
    backend: str,
    device: str,
    *,
    triton_available: Callable[[], bool],
) -> None:
    """Fail early when a requested backend cannot run in this environment."""

    if backend == "torch":
        return
    if backend == "triton" and device != "cuda":
        raise SystemExit(
            "The Triton backend requires CUDA tensors. Use --device cuda on a CUDA host."
        )
    if backend == "triton" and not triton_available():
        raise SystemExit(
            "The Triton backend requires torch, triton, and CUDA. "
            "Install GPU extras with: uv sync --group dev --extra gpu"
        )
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend: {backend}")


def require_torch() -> Any:
    """Import PyTorch or exit with the benchmark setup command."""

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for benchmarks. Install with: uv sync --group dev --extra gpu"
        ) from exc
    return torch


def resolve_device(torch: Any, requested: str) -> str:
    """Resolve an auto/cpu/cuda device request into a concrete device string."""

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return requested


def resolve_dtype(torch: Any, dtype_name: str) -> Any:
    """Resolve a CLI dtype name into a torch dtype."""

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def dtype_label(dtype: Any) -> str:
    """Return a compact label for a torch dtype-like value."""

    return str(dtype).replace("torch.", "")


def correctness_tolerance(dtype: Any) -> tuple[float, float]:
    """Return default rtol/atol values for benchmark correctness checks."""

    label = dtype_label(dtype)
    if label in {"float16", "bfloat16"}:
        return 1e-2, 1e-2
    return 1e-4, 1e-5


def emit_results(
    results: list[BenchmarkResult],
    *,
    args: argparse.Namespace,
    benchmark: str,
    print_table: Callable[[list[BenchmarkResult]], None],
) -> None:
    """Print benchmark results and optionally append reproducible JSONL records."""

    metadata = collect_run_metadata(benchmark, args)
    records = benchmark_records(results, metadata)
    written_path = write_jsonl(records, args.output) if args.output else None

    if args.json:
        print(json.dumps(records, indent=2))
        return

    print_table(results)
    if written_path is not None:
        print(f"Wrote benchmark records to {written_path}")


def print_results_table(
    results: list[BenchmarkResult],
    *,
    name_width: int,
    include_shape: bool,
) -> None:
    """Render benchmark results with the common latency/throughput columns."""

    shape_column = f" {'shape':<16}" if include_shape else ""
    print(
        f"{'name':<{name_width}} {'device':<8} {'dtype':<9}{shape_column} "
        f"{'p50_ms':>10} {'p95_ms':>10} {'p99_ms':>10} {'GB/s':>10} {'TFLOP/s':>10}"
    )
    for result in results:
        shape = f" {_shape_label(result.shape):<16}" if include_shape else ""
        print(
            f"{result.name:<{name_width}} {result.device:<8} {result.dtype:<9}{shape} "
            f"{result.p50_ms:10.4f} {result.p95_ms:10.4f} {result.p99_ms:10.4f} "
            f"{result.bandwidth_gbps:10.2f} {result.tflops:10.4f}"
        )


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)

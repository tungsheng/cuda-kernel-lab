"""Benchmark one-token decode attention across available backends."""

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
from cuda_kernel_lab.ops.attention import (
    decode_attention_flop_count,
    decode_attention_memory_traffic_bytes,
)
from cuda_kernel_lab.optimization import attention_optimization


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)

    results = []
    for backend in selected_backends(args.backend, device, triton_available=triton_is_available):
        ensure_backend_available(backend, device, triton_available=triton_is_available)
        if backend != "torch":
            raise SystemExit("benchmark-attention currently supports only --backend torch.")
        results.append(
            run_one(
                torch=torch,
                backend=backend,
                seq_len=args.seq_len,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                device=device,
                scale=args.scale,
                warmup=args.warmup,
                iterations=args.iterations,
                skip_correctness=args.skip_correctness,
            )
        )

    emit_results(results, args=args, benchmark="attention", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Attention score scale. Default: head_dim ** -0.5.",
    )
    add_common_benchmark_args(parser)
    return parser.parse_args()


def triton_is_available() -> bool:
    return False


def run_one(
    *,
    torch: Any,
    backend: str,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: Any,
    device: str,
    scale: float | None,
    warmup: int,
    iterations: int,
    skip_correctness: bool,
) -> BenchmarkResult:
    if seq_len <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ValueError("seq_len, num_heads, and head_dim must be positive")

    query = torch.randn((num_heads, head_dim), device=device, dtype=dtype)
    key_cache = torch.randn((seq_len, num_heads, head_dim), device=device, dtype=dtype)
    value_cache = torch.randn((seq_len, num_heads, head_dim), device=device, dtype=dtype)
    out = torch.empty_like(query)
    fn = build_op(backend, query, key_cache, value_cache, scale, out)
    effective_scale = scale if scale is not None else head_dim**-0.5

    correctness = None
    if not skip_correctness:
        expected = reference_decode_attention(torch, query, key_cache, value_cache, effective_scale)
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
        f"{backend}:decode_attention",
        fn,
        device=device,
        dtype=dtype_label(dtype),
        shape=(seq_len, num_heads, head_dim),
        bytes_moved=decode_attention_memory_traffic_bytes(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_size=dtype_size,
        ),
        flops=decode_attention_flop_count(
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        warmup=warmup,
        iterations=iterations,
        strategy="torch-baseline" if backend == "torch" else "triton-decode-attention",
        variant=(
            f"seq_len={seq_len}, num_heads={num_heads}, "
            f"head_dim={head_dim}, scale={effective_scale:g}"
        ),
        parameters={
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "scale": effective_scale,
        },
        optimization=attention_optimization(backend=backend),
        correctness=correctness,
    )


def build_op(
    backend: str,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    scale: float | None,
    out: Any | None,
) -> Callable[[], Any]:
    if backend == "torch":
        from cuda_kernel_lab.kernels.torch_baselines import decode_attention

        return lambda: decode_attention(query, key_cache, value_cache, scale=scale, out=out)

    raise ValueError(f"unknown backend: {backend}")


def reference_decode_attention(
    torch: Any,
    query: Any,
    key_cache: Any,
    value_cache: Any,
    scale: float,
) -> Any:
    q = query.float().unsqueeze(1)
    k = key_cache.float().permute(1, 2, 0)
    scores = torch.bmm(q, k).squeeze(1) * scale
    probs = scores.softmax(dim=-1).to(dtype=query.dtype)
    return torch.bmm(probs.unsqueeze(1), value_cache.permute(1, 0, 2)).squeeze(1)


def print_table(results: list[BenchmarkResult]) -> None:
    print_results_table(results, name_width=24, include_shape=True)


if __name__ == "__main__":
    main()

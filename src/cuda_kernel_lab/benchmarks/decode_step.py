"""Benchmark a synthetic one-token decode step with eager and CUDA Graph launch modes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter, process_time
from typing import Any

from cuda_kernel_lab.benchmark import BenchmarkResult, check_tensors_close
from cuda_kernel_lab.benchmark_cli import (
    DTYPES,
    correctness_tolerance,
    dtype_label,
    emit_results,
    require_torch,
    resolve_device,
    resolve_dtype,
)
from cuda_kernel_lab.metrics import dtype_size_bytes, percentile
from cuda_kernel_lab.ops.decode_step import (
    decode_step_flop_count,
    decode_step_memory_traffic_bytes,
)
from cuda_kernel_lab.optimization import decode_step_optimization

MODES = ("naive-eager", "fused-eager", "naive-graph", "fused-graph")
DEFAULT_BATCH_SIZE = 1
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_INTERMEDIATE_DIM = 4096
DEFAULT_SEQ_LEN = 2048
DEFAULT_NUM_HEADS = 16
DEFAULT_HEAD_DIM = 64
DEFAULT_EPS = 1e-6


@dataclass(frozen=True)
class DecodeStepInputs:
    """Static tensors reused by eager and graph replay benchmark modes."""

    x: Any
    rms_weight: Any
    q_weight: Any
    gate_weight: Any
    up_weight: Any
    key_cache: Any
    value_cache: Any


@dataclass(frozen=True)
class TimedLoop:
    """Host/device timing streams for one benchmark loop."""

    host_latencies_ms: list[float]
    device_latencies_ms: list[float]
    cpu_latencies_ms: list[float]


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    fused_available = triton_fused_available()
    modes = selected_modes(args.mode, device=device, fused_available=fused_available)
    validate_shape(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    torch.manual_seed(args.seed)
    inputs = build_inputs(
        torch=torch,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    reference = (
        None
        if args.skip_correctness
        else build_decode_step_op(torch, inputs, "naive", args.eps)()
    )

    results = [
        run_one(
            torch=torch,
            inputs=inputs,
            mode=mode,
            dtype=dtype,
            device=device,
            eps=args.eps,
            warmup=args.warmup,
            iterations=args.iterations,
            reference=reference,
        )
        for mode in modes
    ]
    emit_results(results, args=args, benchmark="decode_step", print_table=print_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--intermediate-dim", type=int, default=DEFAULT_INTERMEDIATE_DIM)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip the pre-benchmark correctness check against naive eager mode.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON records with run metadata.")
    parser.add_argument(
        "--output",
        help="Append JSONL benchmark records to this path, including run metadata.",
    )
    return parser.parse_args()


def triton_fused_available() -> bool:
    try:
        from cuda_kernel_lab.kernels.triton.norms import is_available as norms_available
        from cuda_kernel_lab.kernels.triton.swiglu import is_available as swiglu_available
    except ImportError:
        return False
    return norms_available() and swiglu_available()


def selected_modes(mode: str, *, device: str, fused_available: bool) -> tuple[str, ...]:
    """Return benchmark modes that can run in the current environment."""

    if mode != "all":
        ensure_mode_available(mode, device=device, fused_available=fused_available)
        return (mode,)

    modes = ["naive-eager"]
    if device == "cuda" and fused_available:
        modes.append("fused-eager")
    if device == "cuda":
        modes.append("naive-graph")
        if fused_available:
            modes.append("fused-graph")
    return tuple(modes)


def ensure_mode_available(mode: str, *, device: str, fused_available: bool) -> None:
    kernel_strategy, launch_strategy = split_mode(mode)
    if launch_strategy == "graph" and device != "cuda":
        raise SystemExit("CUDA Graph replay modes require --device cuda.")
    if kernel_strategy == "fused" and device != "cuda":
        raise SystemExit("Fused decode-step modes require --device cuda.")
    if kernel_strategy == "fused" and not fused_available:
        raise SystemExit(
            "Fused decode-step modes require Triton RMSNorm/SwiGLU on a CUDA-capable host."
        )


def split_mode(mode: str) -> tuple[str, str]:
    try:
        kernel_strategy, launch_strategy = mode.split("-", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"unknown decode-step mode: {mode}") from exc
    if kernel_strategy not in {"naive", "fused"} or launch_strategy not in {"eager", "graph"}:
        raise ValueError(f"unknown decode-step mode: {mode}")
    return kernel_strategy, launch_strategy


def validate_shape(
    *,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> None:
    decode_step_memory_traffic_bytes(
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype_size=1,
    )
    attention_dim = num_heads * head_dim
    if hidden_dim != attention_dim:
        raise ValueError("hidden_dim must equal num_heads * head_dim for this synthetic step")


def build_inputs(
    *,
    torch: Any,
    batch_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: Any,
    device: str,
) -> DecodeStepInputs:
    attention_dim = num_heads * head_dim
    scale = 0.02
    return DecodeStepInputs(
        x=torch.randn((batch_size, hidden_dim), device=device, dtype=dtype),
        rms_weight=torch.randn((hidden_dim,), device=device, dtype=dtype),
        q_weight=torch.randn((hidden_dim, attention_dim), device=device, dtype=dtype) * scale,
        gate_weight=torch.randn((hidden_dim, intermediate_dim), device=device, dtype=dtype)
        * scale,
        up_weight=torch.randn((hidden_dim, intermediate_dim), device=device, dtype=dtype) * scale,
        key_cache=torch.randn(
            (batch_size, seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype,
        ),
        value_cache=torch.randn(
            (batch_size, seq_len, num_heads, head_dim),
            device=device,
            dtype=dtype,
        ),
    )


def run_one(
    *,
    torch: Any,
    inputs: DecodeStepInputs,
    mode: str,
    dtype: Any,
    device: str,
    eps: float,
    warmup: int,
    iterations: int,
    reference: Any | None,
) -> BenchmarkResult:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    kernel_strategy, launch_strategy = split_mode(mode)
    ensure_mode_available(
        mode,
        device=device,
        fused_available=(kernel_strategy != "fused" or triton_fused_available()),
    )
    raw_fn = build_decode_step_op(torch, inputs, kernel_strategy, eps)
    benchmark_fn = prepare_launch_callable(
        torch=torch,
        fn=raw_fn,
        launch_strategy=launch_strategy,
        device=device,
        warmup=warmup,
    )
    correctness = None
    if reference is not None:
        actual = benchmark_fn()
        rtol, atol = correctness_tolerance(dtype)
        correctness = check_tensors_close(
            actual,
            reference,
            torch=torch,
            rtol=rtol,
            atol=atol,
        )

    timings = benchmark_timed_loop(
        torch=torch,
        fn=benchmark_fn,
        device=device,
        warmup=warmup,
        iterations=iterations,
    )
    batch_size, hidden_dim = inputs.x.shape
    intermediate_dim = inputs.gate_weight.shape[1]
    seq_len = inputs.key_cache.shape[1]
    num_heads = inputs.key_cache.shape[2]
    head_dim = inputs.key_cache.shape[3]
    dtype_size = dtype_size_bytes(dtype)
    metrics = timing_metrics(
        timings,
        tokens_per_step=batch_size,
        graph_replay=launch_strategy == "graph",
    )

    return BenchmarkResult(
        name=f"{kernel_strategy}:decode_step",
        device=device,
        dtype=dtype_label(dtype),
        shape=(batch_size, seq_len, num_heads, head_dim, intermediate_dim),
        latencies_ms=timings.host_latencies_ms,
        bytes_moved=decode_step_memory_traffic_bytes(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_size=dtype_size,
        ),
        flops=decode_step_flop_count(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        strategy=f"{kernel_strategy}-{launch_strategy}",
        variant=(
            f"mode={mode}, batch_size={batch_size}, seq_len={seq_len}, "
            f"hidden_dim={hidden_dim}, intermediate_dim={intermediate_dim}"
        ),
        parameters={
            "mode": mode,
            "kernel_strategy": kernel_strategy,
            "launch_strategy": launch_strategy,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "eps": eps,
        },
        metrics=metrics,
        optimization=decode_step_optimization(
            kernel_strategy=kernel_strategy,
            launch_strategy=launch_strategy,
        ),
        correctness=correctness,
    )


def build_decode_step_op(
    torch: Any,
    inputs: DecodeStepInputs,
    kernel_strategy: str,
    eps: float,
) -> Callable[[], Any]:
    if kernel_strategy == "naive":
        return lambda: naive_decode_step(torch, inputs, eps)
    if kernel_strategy == "fused":
        return lambda: fused_decode_step(torch, inputs, eps)
    raise ValueError(f"unknown kernel strategy: {kernel_strategy}")


def naive_decode_step(torch: Any, inputs: DecodeStepInputs, eps: float) -> Any:
    squared = inputs.x.pow(2)
    variance = squared.mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normalized = inputs.x * inv_rms
    normalized = normalized * inputs.rms_weight
    q_flat = normalized @ inputs.q_weight
    query = _query_view(q_flat, inputs)
    context = _decode_attention_batched(torch, query, inputs.key_cache, inputs.value_cache)
    gate = normalized @ inputs.gate_weight
    up = normalized @ inputs.up_weight
    sigmoid = gate.sigmoid()
    silu = gate * sigmoid
    ff = silu * up
    return context.flatten(start_dim=1) + ff[:, : q_flat.shape[1]]


def fused_decode_step(torch: Any, inputs: DecodeStepInputs, eps: float) -> Any:
    from cuda_kernel_lab.kernels.triton import rmsnorm, swiglu

    normalized = torch.empty_like(inputs.x)
    rmsnorm(inputs.x, inputs.rms_weight, eps=eps, out=normalized)
    q_flat = normalized @ inputs.q_weight
    query = _query_view(q_flat, inputs)
    context = _decode_attention_batched(torch, query, inputs.key_cache, inputs.value_cache)
    gate = normalized @ inputs.gate_weight
    up = normalized @ inputs.up_weight
    ff = swiglu(gate, up)
    return context.flatten(start_dim=1) + ff[:, : q_flat.shape[1]]


def _query_view(q_flat: Any, inputs: DecodeStepInputs) -> Any:
    batch_size = inputs.x.shape[0]
    num_heads = inputs.key_cache.shape[2]
    head_dim = inputs.key_cache.shape[3]
    return q_flat.reshape(batch_size, num_heads, head_dim)


def _decode_attention_batched(torch: Any, query: Any, key_cache: Any, value_cache: Any) -> Any:
    scale = query.shape[-1] ** -0.5
    scores = torch.einsum("bhd,bshd->bhs", query.float(), key_cache.float()) * scale
    probs = scores.softmax(dim=-1).to(dtype=query.dtype)
    return torch.einsum("bhs,bshd->bhd", probs, value_cache)


def prepare_launch_callable(
    *,
    torch: Any,
    fn: Callable[[], Any],
    launch_strategy: str,
    device: str,
    warmup: int,
) -> Callable[[], Any]:
    if launch_strategy == "eager":
        return fn
    if launch_strategy != "graph":
        raise ValueError(f"unknown launch strategy: {launch_strategy}")
    if device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA Graph replay modes require --device cuda.")

    for _ in range(max(1, min(warmup, 3))):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn()

    def replay() -> Any:
        graph.replay()
        return graph_output

    return replay


def benchmark_timed_loop(
    *,
    torch: Any,
    fn: Callable[[], Any],
    device: str,
    warmup: int,
    iterations: int,
) -> TimedLoop:
    for _ in range(warmup):
        fn()
    _synchronize(torch, device)

    host_latencies_ms: list[float] = []
    device_latencies_ms: list[float] = []
    cpu_latencies_ms: list[float] = []
    use_cuda_events = device == "cuda" and torch.cuda.is_available()
    for _ in range(iterations):
        host_start = perf_counter()
        cpu_start = process_time()
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            device_ms = float(start.elapsed_time(end))
        else:
            fn()
            _synchronize(torch, device)
            device_ms = (perf_counter() - host_start) * 1_000
        cpu_ms = (process_time() - cpu_start) * 1_000
        host_ms = (perf_counter() - host_start) * 1_000
        host_latencies_ms.append(host_ms)
        device_latencies_ms.append(device_ms)
        cpu_latencies_ms.append(cpu_ms)

    return TimedLoop(
        host_latencies_ms=host_latencies_ms,
        device_latencies_ms=device_latencies_ms,
        cpu_latencies_ms=cpu_latencies_ms,
    )


def timing_metrics(
    timings: TimedLoop,
    *,
    tokens_per_step: int,
    graph_replay: bool,
) -> dict[str, float | int | bool]:
    host_p50 = percentile(timings.host_latencies_ms, 50)
    device_p50 = percentile(timings.device_latencies_ms, 50)
    launch_overheads = [
        max(host_ms - device_ms, 0.0)
        for host_ms, device_ms in zip(
            timings.host_latencies_ms,
            timings.device_latencies_ms,
            strict=True,
        )
    ]
    total_host_s = sum(timings.host_latencies_ms) / 1_000
    total_cpu_s = sum(timings.cpu_latencies_ms) / 1_000
    cpu_utilization = (total_cpu_s / total_host_s * 100) if total_host_s > 0 else 0.0
    tokens_per_second = tokens_per_step / (host_p50 / 1_000) if host_p50 > 0 else 0.0
    return {
        "host_p50_ms": host_p50,
        "device_p50_ms": device_p50,
        "device_p95_ms": percentile(timings.device_latencies_ms, 95),
        "device_p99_ms": percentile(timings.device_latencies_ms, 99),
        "launch_overhead_p50_ms": percentile(launch_overheads, 50),
        "cpu_utilization_pct": cpu_utilization,
        "tokens_per_second_p50": tokens_per_second,
        "tokens_per_step": tokens_per_step,
        "graph_replay": graph_replay,
    }


def _synchronize(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def print_table(results: list[BenchmarkResult]) -> None:
    print(
        f"{'mode':<14} {'device':<8} {'dtype':<9} {'shape':<24} "
        f"{'host_p50':>10} {'device_p50':>10} {'launch_p50':>11} "
        f"{'tok/s':>10} {'cpu_%':>8} {'GB/s':>10} {'TFLOP/s':>10}"
    )
    for result in results:
        metrics = result.metrics or {}
        print(
            f"{result.strategy:<14} {result.device:<8} {result.dtype:<9} "
            f"{_shape_label(result.shape):<24} "
            f"{result.p50_ms:10.4f} {_metric(metrics, 'device_p50_ms'):10.4f} "
            f"{_metric(metrics, 'launch_overhead_p50_ms'):11.4f} "
            f"{_metric(metrics, 'tokens_per_second_p50'):10.2f} "
            f"{_metric(metrics, 'cpu_utilization_pct'):8.2f} "
            f"{result.bandwidth_gbps:10.2f} {result.tflops:10.4f}"
        )


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


if __name__ == "__main__":
    main()

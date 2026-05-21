"""Benchmark a synthetic one-token decode step with eager and CUDA Graph launch modes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
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

STATIC_MODES = (
    "naive-eager",
    "fused-eager",
    "naive-graph",
    "fused-graph",
    "fused-piecewise-graph",
)
DYNAMIC_MODES = ("dynamic-eager", "dynamic-piecewise-graph")
MODES = (*STATIC_MODES, *DYNAMIC_MODES)
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_INTERMEDIATE_DIM = 4096
DEFAULT_SEQ_LEN = 2048
DEFAULT_MIN_SEQ_LEN = 128
DEFAULT_NUM_HEADS = 16
DEFAULT_HEAD_DIM = 64
DEFAULT_EPS = 1e-6
DEFAULT_BATCH_BUCKETS = (1, 2, 4, 8)
DEFAULT_PREFILL_INTERVAL = 16
DEFAULT_MIXED_INTERVAL = 7
PHASES = ("decode", "prefill", "mixed")


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


@dataclass(frozen=True)
class TraceStep:
    """One synthetic scheduler decision in a dynamic batching trace."""

    active_batch_size: int
    seq_len: int
    phase: str
    queue_wait_ms: float


@dataclass(frozen=True)
class DynamicTimedLoop:
    """Timing and scheduling streams for a dynamic trace replay."""

    timings: TimedLoop
    scheduler_cpu_latencies_ms: list[float]
    graph_hits: int
    recapture_count: int


@dataclass
class PiecewiseGraphRuntime:
    """Captured pre/post graph regions with eager attention in the middle."""

    torch: Any
    inputs: DecodeStepInputs
    q_flat: Any
    context_flat: Any
    output: Any
    pre_graph: Any
    post_graph: Any

    def replay(self, *, active_batch_size: int | None = None, seq_len: int | None = None) -> Any:
        self.pre_graph.replay()
        active = active_batch_size or self.inputs.x.shape[0]
        length = seq_len or self.inputs.key_cache.shape[1]
        query = _query_view(self.q_flat[:active], self.inputs)
        context = _decode_attention_batched(
            self.torch,
            query,
            self.inputs.key_cache[:active, :length],
            self.inputs.value_cache[:active, :length],
        )
        self.context_flat[:active].copy_(context.flatten(start_dim=1))
        self.post_graph.replay()
        return self.output[:active]


def main() -> None:
    args = parse_args()
    torch = require_torch()
    device = resolve_device(torch, args.device)
    dtype = resolve_dtype(torch, args.dtype)
    fused_available = triton_fused_available()
    dynamic_trace = args.dynamic_trace or args.mode in DYNAMIC_MODES
    modes = selected_modes(
        args.mode,
        device=device,
        fused_available=fused_available,
        dynamic_trace=dynamic_trace,
    )
    batch_size = args.max_batch_size if dynamic_trace else args.batch_size
    validate_shape(
        batch_size=batch_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
    )

    torch.manual_seed(args.seed)
    inputs = build_inputs(
        torch=torch,
        batch_size=batch_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        seq_len=args.seq_len,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )
    if dynamic_trace:
        trace = generate_dynamic_trace(
            steps=args.iterations,
            max_batch_size=args.max_batch_size,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.seq_len,
            seed=args.seed,
            prefill_interval=args.prefill_interval,
            mixed_interval=args.mixed_interval,
        )
        batch_buckets = parse_batch_buckets(args.batch_buckets, max_batch_size=args.max_batch_size)
        results = [
            run_dynamic_trace(
                torch=torch,
                inputs=inputs,
                mode=mode,
                trace=trace,
                batch_buckets=batch_buckets,
                dtype=dtype,
                device=device,
                eps=args.eps,
                warmup=args.warmup,
                reference_checks=not args.skip_correctness,
            )
            for mode in modes
        ]
        emit_results(results, args=args, benchmark="decode_step", print_table=print_table)
        return

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
    parser.add_argument("--max-batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--intermediate-dim", type=int, default=DEFAULT_INTERMEDIATE_DIM)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--min-seq-len", type=int, default=DEFAULT_MIN_SEQ_LEN)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--dynamic-trace",
        action="store_true",
        help="Replay a synthetic dynamic batching/scheduling trace.",
    )
    parser.add_argument(
        "--batch-buckets",
        default=",".join(str(bucket) for bucket in DEFAULT_BATCH_BUCKETS),
        help="Comma-separated batch buckets for dynamic piecewise graph replay.",
    )
    parser.add_argument(
        "--prefill-interval",
        type=int,
        default=DEFAULT_PREFILL_INTERVAL,
        help="Every Nth dynamic trace step is labeled prefill. Use 0 to disable.",
    )
    parser.add_argument(
        "--mixed-interval",
        type=int,
        default=DEFAULT_MIXED_INTERVAL,
        help="Every Nth dynamic trace step is labeled mixed. Use 0 to disable.",
    )
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


def selected_modes(
    mode: str,
    *,
    device: str,
    fused_available: bool,
    dynamic_trace: bool = False,
) -> tuple[str, ...]:
    """Return benchmark modes that can run in the current environment."""

    if mode != "all":
        ensure_mode_available(mode, device=device, fused_available=fused_available)
        return (mode,)

    if dynamic_trace:
        modes = ["dynamic-eager"]
        if device == "cuda" and fused_available:
            modes.append("dynamic-piecewise-graph")
        return tuple(modes)

    modes = ["naive-eager"]
    if device == "cuda" and fused_available:
        modes.append("fused-eager")
    if device == "cuda":
        modes.append("naive-graph")
        if fused_available:
            modes.append("fused-graph")
            modes.append("fused-piecewise-graph")
    return tuple(modes)


def ensure_mode_available(mode: str, *, device: str, fused_available: bool) -> None:
    if mode == "dynamic-eager":
        return
    if mode == "dynamic-piecewise-graph":
        if device != "cuda":
            raise SystemExit("Dynamic piecewise CUDA Graph replay requires --device cuda.")
        if not fused_available:
            raise SystemExit(
                "Dynamic piecewise CUDA Graph replay requires Triton RMSNorm/SwiGLU."
            )
        return

    kernel_strategy, launch_strategy = split_mode(mode)
    if launch_strategy in {"graph", "piecewise_graph"} and device != "cuda":
        raise SystemExit("CUDA Graph replay modes require --device cuda.")
    if kernel_strategy == "fused" and device != "cuda":
        raise SystemExit("Fused decode-step modes require --device cuda.")
    if kernel_strategy == "fused" and not fused_available:
        raise SystemExit(
            "Fused decode-step modes require Triton RMSNorm/SwiGLU on a CUDA-capable host."
        )


def split_mode(mode: str) -> tuple[str, str]:
    if mode == "fused-piecewise-graph":
        return "fused", "piecewise_graph"
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


def parse_batch_buckets(value: str, *, max_batch_size: int) -> tuple[int, ...]:
    """Parse and normalize dynamic graph batch buckets."""

    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    buckets = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not buckets:
        raise ValueError("batch_buckets must include at least one bucket")
    if buckets[0] <= 0:
        raise ValueError("batch_buckets must be positive")
    if buckets[-1] < max_batch_size:
        buckets.append(max_batch_size)
    return tuple(buckets)


def generate_dynamic_trace(
    *,
    steps: int,
    max_batch_size: int,
    min_seq_len: int,
    max_seq_len: int,
    seed: int,
    prefill_interval: int,
    mixed_interval: int,
) -> tuple[TraceStep, ...]:
    """Build a deterministic synthetic dynamic batching trace."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    if min_seq_len <= 0:
        raise ValueError("min_seq_len must be positive")
    if max_seq_len < min_seq_len:
        raise ValueError("max_seq_len must be greater than or equal to min_seq_len")
    rng = Random(seed)
    trace = []
    for index in range(steps):
        active_batch_size = rng.randint(1, max_batch_size)
        seq_len = rng.randint(min_seq_len, max_seq_len)
        phase = _trace_phase(
            index,
            prefill_interval=prefill_interval,
            mixed_interval=mixed_interval,
        )
        queue_wait_ms = _synthetic_queue_wait_ms(
            active_batch_size=active_batch_size,
            max_batch_size=max_batch_size,
            phase=phase,
        )
        trace.append(
            TraceStep(
                active_batch_size=active_batch_size,
                seq_len=seq_len,
                phase=phase,
                queue_wait_ms=queue_wait_ms,
            )
        )
    return tuple(trace)


def _trace_phase(index: int, *, prefill_interval: int, mixed_interval: int) -> str:
    step_number = index + 1
    if mixed_interval > 0 and step_number % mixed_interval == 0:
        return "mixed"
    if prefill_interval > 0 and step_number % prefill_interval == 0:
        return "prefill"
    return "decode"


def _synthetic_queue_wait_ms(
    *,
    active_batch_size: int,
    max_batch_size: int,
    phase: str,
) -> float:
    fill_wait = (max_batch_size - active_batch_size) * 0.01
    phase_wait = {"decode": 0.0, "mixed": 0.03, "prefill": 0.06}[phase]
    return fill_wait + phase_wait


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
    if launch_strategy == "piecewise_graph":
        runtime = prepare_piecewise_graph_runtime(
            torch=torch,
            inputs=inputs,
            eps=eps,
            warmup=warmup,
        )
        benchmark_fn = runtime.replay
    else:
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
        graph_replay=launch_strategy in {"graph", "piecewise_graph"},
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


def run_dynamic_trace(
    *,
    torch: Any,
    inputs: DecodeStepInputs,
    mode: str,
    trace: tuple[TraceStep, ...],
    batch_buckets: tuple[int, ...],
    dtype: Any,
    device: str,
    eps: float,
    warmup: int,
    reference_checks: bool,
) -> BenchmarkResult:
    """Replay a synthetic dynamic batching trace."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not trace:
        raise ValueError("trace must include at least one step")
    if mode not in DYNAMIC_MODES:
        raise ValueError(f"unknown dynamic decode-step mode: {mode}")

    kernel_strategy = "fused" if device == "cuda" and triton_fused_available() else "naive"
    if mode == "dynamic-piecewise-graph":
        ensure_mode_available(mode, device=device, fused_available=triton_fused_available())
        graph_cache = build_piecewise_graph_cache(
            torch=torch,
            inputs=inputs,
            batch_buckets=batch_buckets,
            eps=eps,
            warmup=warmup,
        )

        def run_step(step: TraceStep) -> bool:
            bucket = choose_batch_bucket(step.active_batch_size, batch_buckets)
            runtime = graph_cache[bucket]
            copy_trace_step_inputs(inputs, runtime.inputs, step=step)
            runtime.replay(active_batch_size=step.active_batch_size, seq_len=step.seq_len)
            return True

        graph_replay = True
        launch_strategy = "piecewise_graph"
    else:

        def run_step(step: TraceStep) -> bool:
            step_inputs = slice_trace_step_inputs(inputs, step=step)
            fn = build_decode_step_op(torch, step_inputs, kernel_strategy, eps)
            fn()
            return False

        graph_replay = False
        launch_strategy = "eager"

    correctness = None
    if reference_checks and mode == "dynamic-piecewise-graph":
        correctness = check_dynamic_piecewise_correctness(
            torch=torch,
            inputs=inputs,
            graph_cache=graph_cache,
            trace=trace,
            batch_buckets=batch_buckets,
            eps=eps,
            dtype=dtype,
            kernel_strategy=kernel_strategy,
        )

    dynamic_loop = benchmark_dynamic_timed_loop(
        torch=torch,
        run_step=run_step,
        trace=trace,
        device=device,
        warmup=warmup,
    )
    batch_size, hidden_dim = inputs.x.shape
    intermediate_dim = inputs.gate_weight.shape[1]
    max_seq_len = inputs.key_cache.shape[1]
    num_heads = inputs.key_cache.shape[2]
    head_dim = inputs.key_cache.shape[3]
    dtype_size = dtype_size_bytes(dtype)
    metrics = dynamic_timing_metrics(
        dynamic_loop,
        trace=trace,
        batch_buckets=batch_buckets,
        max_batch_size=batch_size,
        graph_replay=graph_replay,
    )

    return BenchmarkResult(
        name=f"{mode}:decode_step",
        device=device,
        dtype=dtype_label(dtype),
        shape=(batch_size, max_seq_len, num_heads, head_dim, intermediate_dim),
        latencies_ms=dynamic_loop.timings.host_latencies_ms,
        bytes_moved=average_trace_memory_traffic_bytes(
            trace,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_size=dtype_size,
        ),
        flops=average_trace_flop_count(
            trace,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        strategy=mode,
        variant=(
            f"mode={mode}, max_batch_size={batch_size}, seq_len={max_seq_len}, "
            f"buckets={','.join(str(bucket) for bucket in batch_buckets)}"
        ),
        parameters={
            "mode": mode,
            "kernel_strategy": kernel_strategy,
            "launch_strategy": launch_strategy,
            "max_batch_size": batch_size,
            "batch_buckets": batch_buckets,
            "hidden_dim": hidden_dim,
            "intermediate_dim": intermediate_dim,
            "seq_len": max_seq_len,
            "min_seq_len": min(step.seq_len for step in trace),
            "num_heads": num_heads,
            "head_dim": head_dim,
            "trace_steps": len(trace),
            "eps": eps,
        },
        metrics=metrics,
        optimization=decode_step_optimization(
            kernel_strategy=kernel_strategy,
            launch_strategy=launch_strategy,
        ),
        correctness=correctness,
    )


def build_piecewise_graph_cache(
    *,
    torch: Any,
    inputs: DecodeStepInputs,
    batch_buckets: tuple[int, ...],
    eps: float,
    warmup: int,
) -> dict[int, PiecewiseGraphRuntime]:
    """Capture one piecewise graph runtime per dynamic batch bucket."""

    cache = {}
    for bucket in batch_buckets:
        bucket_inputs = DecodeStepInputs(
            x=torch.empty_like(inputs.x[:bucket]),
            rms_weight=inputs.rms_weight,
            q_weight=inputs.q_weight,
            gate_weight=inputs.gate_weight,
            up_weight=inputs.up_weight,
            key_cache=torch.empty_like(inputs.key_cache[:bucket]),
            value_cache=torch.empty_like(inputs.value_cache[:bucket]),
        )
        bucket_inputs.x.copy_(inputs.x[:bucket])
        bucket_inputs.key_cache.copy_(inputs.key_cache[:bucket])
        bucket_inputs.value_cache.copy_(inputs.value_cache[:bucket])
        cache[bucket] = prepare_piecewise_graph_runtime(
            torch=torch,
            inputs=bucket_inputs,
            eps=eps,
            warmup=warmup,
        )
    return cache


def choose_batch_bucket(active_batch_size: int, batch_buckets: tuple[int, ...]) -> int:
    """Return the smallest configured bucket that can host active_batch_size."""

    for bucket in batch_buckets:
        if active_batch_size <= bucket:
            return bucket
    raise ValueError(
        f"active batch size {active_batch_size} exceeds max bucket {batch_buckets[-1]}"
    )


def slice_trace_step_inputs(inputs: DecodeStepInputs, *, step: TraceStep) -> DecodeStepInputs:
    """Return variable-shape views for one dynamic eager trace step."""

    active = step.active_batch_size
    length = step.seq_len
    return DecodeStepInputs(
        x=inputs.x[:active],
        rms_weight=inputs.rms_weight,
        q_weight=inputs.q_weight,
        gate_weight=inputs.gate_weight,
        up_weight=inputs.up_weight,
        key_cache=inputs.key_cache[:active, :length],
        value_cache=inputs.value_cache[:active, :length],
    )


def copy_trace_step_inputs(
    source: DecodeStepInputs,
    target: DecodeStepInputs,
    *,
    step: TraceStep,
) -> None:
    """Copy active trace tensors into static bucket buffers before graph replay."""

    active = step.active_batch_size
    length = step.seq_len
    target.x[:active].copy_(source.x[:active])
    target.key_cache[:active, :length].copy_(source.key_cache[:active, :length])
    target.value_cache[:active, :length].copy_(source.value_cache[:active, :length])


def check_dynamic_piecewise_correctness(
    *,
    torch: Any,
    inputs: DecodeStepInputs,
    graph_cache: dict[int, PiecewiseGraphRuntime],
    trace: tuple[TraceStep, ...],
    batch_buckets: tuple[int, ...],
    eps: float,
    dtype: Any,
    kernel_strategy: str,
) -> Any:
    """Compare the first dynamic piecewise replay against the matching eager path."""

    step = trace[0]
    bucket = choose_batch_bucket(step.active_batch_size, batch_buckets)
    runtime = graph_cache[bucket]
    copy_trace_step_inputs(inputs, runtime.inputs, step=step)
    actual = runtime.replay(active_batch_size=step.active_batch_size, seq_len=step.seq_len)
    reference_inputs = slice_trace_step_inputs(inputs, step=step)
    reference = build_decode_step_op(torch, reference_inputs, kernel_strategy, eps)()
    rtol, atol = correctness_tolerance(dtype)
    return check_tensors_close(
        actual,
        reference,
        torch=torch,
        rtol=rtol,
        atol=atol,
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
    batch_size = q_flat.shape[0]
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


def prepare_piecewise_graph_runtime(
    *,
    torch: Any,
    inputs: DecodeStepInputs,
    eps: float,
    warmup: int,
) -> PiecewiseGraphRuntime:
    """Capture static fused pre/post regions around eager dynamic attention."""

    if not torch.cuda.is_available():
        raise SystemExit("Piecewise CUDA Graph replay requires a CUDA-capable PyTorch runtime.")

    from cuda_kernel_lab.kernels.triton import rmsnorm, swiglu

    batch_size = inputs.x.shape[0]
    attention_dim = inputs.q_weight.shape[1]
    intermediate_dim = inputs.gate_weight.shape[1]
    normalized = torch.empty_like(inputs.x)
    q_flat = torch.empty((batch_size, attention_dim), device=inputs.x.device, dtype=inputs.x.dtype)
    gate = torch.empty(
        (batch_size, intermediate_dim),
        device=inputs.x.device,
        dtype=inputs.x.dtype,
    )
    up = torch.empty_like(gate)
    ff = torch.empty_like(gate)
    context_flat = torch.empty_like(q_flat)
    output = torch.empty_like(q_flat)

    def pre_region() -> None:
        rmsnorm(inputs.x, inputs.rms_weight, eps=eps, out=normalized)
        torch.mm(normalized, inputs.q_weight, out=q_flat)
        torch.mm(normalized, inputs.gate_weight, out=gate)
        torch.mm(normalized, inputs.up_weight, out=up)
        swiglu(gate, up, out=ff)

    def post_region() -> None:
        torch.add(context_flat, ff[:, :attention_dim], out=output)

    for _ in range(max(1, min(warmup, 3))):
        pre_region()
        context = _decode_attention_batched(
            torch,
            _query_view(q_flat, inputs),
            inputs.key_cache,
            inputs.value_cache,
        )
        context_flat.copy_(context.flatten(start_dim=1))
        post_region()
    torch.cuda.synchronize()

    pre_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pre_graph):
        pre_region()
    post_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(post_graph):
        post_region()

    return PiecewiseGraphRuntime(
        torch=torch,
        inputs=inputs,
        q_flat=q_flat,
        context_flat=context_flat,
        output=output,
        pre_graph=pre_graph,
        post_graph=post_graph,
    )


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


def benchmark_dynamic_timed_loop(
    *,
    torch: Any,
    run_step: Callable[[TraceStep], bool],
    trace: tuple[TraceStep, ...],
    device: str,
    warmup: int,
) -> DynamicTimedLoop:
    """Measure one synthetic scheduler trace step by step."""

    for step in trace[:warmup]:
        run_step(step)
    _synchronize(torch, device)

    host_latencies_ms: list[float] = []
    device_latencies_ms: list[float] = []
    cpu_latencies_ms: list[float] = []
    scheduler_cpu_latencies_ms: list[float] = []
    graph_hits = 0
    use_cuda_events = device == "cuda" and torch.cuda.is_available()
    for step in trace:
        scheduler_cpu_start = process_time()
        host_start = perf_counter()
        cpu_start = process_time()
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            hit = run_step(step)
            end.record()
            end.synchronize()
            device_ms = float(start.elapsed_time(end))
        else:
            hit = run_step(step)
            _synchronize(torch, device)
            device_ms = (perf_counter() - host_start) * 1_000
        cpu_ms = (process_time() - cpu_start) * 1_000
        host_ms = (perf_counter() - host_start) * 1_000
        scheduler_cpu_ms = (process_time() - scheduler_cpu_start) * 1_000
        host_latencies_ms.append(host_ms)
        device_latencies_ms.append(device_ms)
        cpu_latencies_ms.append(cpu_ms)
        scheduler_cpu_latencies_ms.append(scheduler_cpu_ms)
        graph_hits += int(hit)

    return DynamicTimedLoop(
        timings=TimedLoop(
            host_latencies_ms=host_latencies_ms,
            device_latencies_ms=device_latencies_ms,
            cpu_latencies_ms=cpu_latencies_ms,
        ),
        scheduler_cpu_latencies_ms=scheduler_cpu_latencies_ms,
        graph_hits=graph_hits,
        recapture_count=0,
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


def dynamic_timing_metrics(
    dynamic_loop: DynamicTimedLoop,
    *,
    trace: tuple[TraceStep, ...],
    batch_buckets: tuple[int, ...],
    max_batch_size: int,
    graph_replay: bool,
) -> dict[str, float | int | bool]:
    """Return timing plus synthetic scheduler metrics for a dynamic trace."""

    base = timing_metrics(
        dynamic_loop.timings,
        tokens_per_step=round(sum(step.active_batch_size for step in trace) / len(trace)),
        graph_replay=graph_replay,
    )
    total_host_s = sum(dynamic_loop.timings.host_latencies_ms) / 1_000
    total_tokens = sum(step.active_batch_size for step in trace)
    padded_tokens = (
        sum(choose_batch_bucket(step.active_batch_size, batch_buckets) for step in trace)
        if graph_replay
        else total_tokens
    )
    phase_counts = {phase: sum(1 for step in trace if step.phase == phase) for phase in PHASES}
    base.update(
        {
            "tokens_per_second": total_tokens / total_host_s if total_host_s > 0 else 0.0,
            "tokens_per_step_avg": total_tokens / len(trace),
            "graph_hit_rate_pct": dynamic_loop.graph_hits / len(trace) * 100,
            "padding_waste_pct": (
                (padded_tokens - total_tokens) / padded_tokens * 100 if padded_tokens else 0.0
            ),
            "recapture_count": dynamic_loop.recapture_count,
            "scheduler_cpu_time_ms": sum(dynamic_loop.scheduler_cpu_latencies_ms),
            "scheduler_cpu_p50_us": percentile(dynamic_loop.scheduler_cpu_latencies_ms, 50)
            * 1_000,
            "queue_wait_p50_ms": percentile([step.queue_wait_ms for step in trace], 50),
            "queue_wait_p95_ms": percentile([step.queue_wait_ms for step in trace], 95),
            "batch_occupancy_avg_pct": total_tokens / (len(trace) * max_batch_size) * 100,
            "decode_steps": phase_counts["decode"],
            "prefill_steps": phase_counts["prefill"],
            "mixed_steps": phase_counts["mixed"],
        }
    )
    base["tokens_per_second_p50"] = base["tokens_per_second"]
    return base


def average_trace_memory_traffic_bytes(
    trace: tuple[TraceStep, ...],
    *,
    hidden_dim: int,
    intermediate_dim: int,
    num_heads: int,
    head_dim: int,
    dtype_size: int,
) -> int:
    values = [
        decode_step_memory_traffic_bytes(
            batch_size=step.active_batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            seq_len=step.seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_size=dtype_size,
        )
        for step in trace
    ]
    return round(sum(values) / len(values))


def average_trace_flop_count(
    trace: tuple[TraceStep, ...],
    *,
    hidden_dim: int,
    intermediate_dim: int,
    num_heads: int,
    head_dim: int,
) -> int:
    values = [
        decode_step_flop_count(
            batch_size=step.active_batch_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            seq_len=step.seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        for step in trace
    ]
    return round(sum(values) / len(values))


def _synchronize(torch: Any, device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def print_table(results: list[BenchmarkResult]) -> None:
    print(
        f"{'mode':<24} {'device':<8} {'dtype':<9} {'shape':<24} "
        f"{'host_p50':>10} {'device_p50':>10} {'launch_p50':>11} "
        f"{'tok/s':>10} {'cpu_%':>8} {'hit_%':>8} {'pad_%':>8} "
        f"{'GB/s':>10} {'TFLOP/s':>10}"
    )
    for result in results:
        metrics = result.metrics or {}
        print(
            f"{result.strategy:<24} {result.device:<8} {result.dtype:<9} "
            f"{_shape_label(result.shape):<24} "
            f"{result.p50_ms:10.4f} {_metric(metrics, 'device_p50_ms'):10.4f} "
            f"{_metric(metrics, 'launch_overhead_p50_ms'):11.4f} "
            f"{_metric(metrics, 'tokens_per_second_p50'):10.2f} "
            f"{_metric(metrics, 'cpu_utilization_pct'):8.2f} "
            f"{_metric(metrics, 'graph_hit_rate_pct'):8.2f} "
            f"{_metric(metrics, 'padding_waste_pct'):8.2f} "
            f"{result.bandwidth_gbps:10.2f} {result.tflops:10.4f}"
        )


def _metric(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


if __name__ == "__main__":
    main()

from __future__ import annotations

import pytest

try:
    import torch
except ImportError:
    torch = None

from cuda_kernel_lab.benchmarks.decode_step import (
    DynamicTimedLoop,
    TimedLoop,
    TraceStep,
    average_trace_flop_count,
    average_trace_memory_traffic_bytes,
    dynamic_timing_metrics,
    generate_dynamic_trace,
    parse_batch_buckets,
    representative_trace_steps,
    run_one,
    selected_modes,
    timing_metrics,
)
from cuda_kernel_lab.ops.decode_step import (
    decode_step_flop_count,
    decode_step_memory_traffic_bytes,
)
from cuda_kernel_lab.optimization import technique_from_strategy

requires_torch = pytest.mark.skipif(torch is None, reason="torch is not installed")


def test_selected_modes_returns_environment_supported_progression() -> None:
    assert selected_modes("all", device="cpu", fused_available=False) == ("naive-eager",)
    assert selected_modes("all", device="cpu", fused_available=True) == ("naive-eager",)
    assert selected_modes("all", device="cuda", fused_available=True) == (
        "naive-eager",
        "fused-eager",
        "naive-graph",
        "fused-graph",
        "fused-piecewise-graph",
        "fused-piecewise-graph-same-stream",
    )
    assert selected_modes(
        "all",
        device="cuda",
        fused_available=True,
        dynamic_trace=True,
    ) == (
        "dynamic-eager",
        "dynamic-piecewise-graph-same-stream",
        "dynamic-piecewise-graph",
    )


def test_selected_modes_rejects_unavailable_graph_mode() -> None:
    with pytest.raises(SystemExit, match="CUDA Graph"):
        selected_modes("naive-graph", device="cpu", fused_available=True)


def test_selected_modes_rejects_fused_cpu_mode() -> None:
    with pytest.raises(SystemExit, match="--device cuda"):
        selected_modes("fused-eager", device="cpu", fused_available=True)


def test_selected_modes_rejects_unavailable_fused_mode() -> None:
    with pytest.raises(SystemExit, match="Triton"):
        selected_modes("fused-eager", device="cuda", fused_available=False)


def test_timing_metrics_reports_tokens_cpu_and_launch_overhead() -> None:
    metrics = timing_metrics(
        TimedLoop(
            host_latencies_ms=[1.0, 2.0, 3.0],
            device_latencies_ms=[0.4, 0.8, 1.2],
            cpu_latencies_ms=[0.2, 0.4, 0.6],
        ),
        tokens_per_step=4,
        graph_replay=True,
    )

    assert metrics["host_p50_ms"] == 2.0
    assert metrics["device_p50_ms"] == 0.8
    assert metrics["launch_overhead_p50_ms"] == pytest.approx(1.2)
    assert metrics["tokens_per_second_p50"] == 2000.0
    assert metrics["cpu_utilization_pct"] == pytest.approx(20.0)
    assert metrics["graph_replay"] is True


def test_decode_step_strategy_labels_map_to_optimization_metadata() -> None:
    optimization = technique_from_strategy("fused-piecewise-graph")

    assert optimization is not None
    assert optimization.method_family == "launch replay"
    assert optimization.method_id == "decode_step.fused_piecewise_graph"

    same_stream = technique_from_strategy("dynamic-piecewise-graph-same-stream")
    assert same_stream is not None
    assert same_stream.method_id == "decode_step.fused_piecewise_graph_same_stream"


def test_dynamic_trace_reports_scheduler_and_padding_metrics() -> None:
    trace = (
        TraceStep(active_batch_size=1, seq_len=2, phase="decode", queue_wait_ms=0.03),
        TraceStep(active_batch_size=3, seq_len=4, phase="mixed", queue_wait_ms=0.01),
    )
    metrics = dynamic_timing_metrics(
        dynamic_loop=DynamicTimedLoop(
            timings=TimedLoop(
                host_latencies_ms=[1.0, 3.0],
                device_latencies_ms=[0.5, 1.0],
                cpu_latencies_ms=[0.2, 0.4],
            ),
            scheduler_cpu_latencies_ms=[0.01, 0.03],
            graph_hits=2,
            recapture_count=0,
            region_latencies_ms={
                "scheduler_decision_host_ms": [0.001, 0.003],
            },
        ),
        trace=trace,
        batch_buckets=(1, 4),
        max_batch_size=4,
        graph_replay=True,
    )

    assert metrics["tokens_per_second"] == pytest.approx(1000.0)
    assert metrics["graph_hit_rate_pct"] == 100.0
    assert metrics["padding_waste_pct"] == pytest.approx(20.0)
    assert metrics["batch_occupancy_avg_pct"] == pytest.approx(50.0)
    assert metrics["host_tail_ratio_p95_p50"] == pytest.approx(1.45)
    assert metrics["host_step_cpu_p95_us"] == pytest.approx(29.0)
    assert metrics["scheduler_cpu_p95_us"] == pytest.approx(29.0)
    assert metrics["scheduler_decision_p95_us"] == pytest.approx(2.9)
    assert metrics["seq_len_min"] == 2
    assert metrics["seq_len_p50"] == 3.0
    assert metrics["seq_len_p95"] == pytest.approx(3.9)
    assert metrics["seq_len_max"] == 4
    assert metrics["decode_steps"] == 1
    assert metrics["mixed_steps"] == 1
    phase_breakdown = metrics["phase_breakdown"]
    assert phase_breakdown["decode"]["steps"] == 1
    assert phase_breakdown["mixed"]["active_batch_avg"] == 3.0
    bucket_breakdown = metrics["bucket_breakdown"]
    assert bucket_breakdown["1"]["steps"] == 1
    assert bucket_breakdown["4"]["padding_waste_pct"] == pytest.approx(25.0)
    assert bucket_breakdown["4"]["host_p99_ms"] == 3.0

    eager_metrics = dynamic_timing_metrics(
        dynamic_loop=DynamicTimedLoop(
            timings=TimedLoop(
                host_latencies_ms=[1.0, 3.0],
                device_latencies_ms=[0.5, 1.0],
                cpu_latencies_ms=[0.2, 0.4],
            ),
            scheduler_cpu_latencies_ms=[0.01, 0.03],
            graph_hits=0,
            recapture_count=0,
            region_latencies_ms={
                "eager_build_host_ms": [0.01, 0.02],
                "eager_run_host_ms": [0.9, 2.7],
            },
        ),
        trace=trace,
        batch_buckets=(1, 4),
        max_batch_size=4,
        graph_replay=False,
    )
    assert eager_metrics["bucket_breakdown"]["4"]["padding_waste_pct"] == 0.0
    orchestration = eager_metrics["orchestration_breakdown"]
    assert orchestration["eager_run_host_ms"]["host_p50_ms"] == pytest.approx(1.8)


def test_dynamic_trace_generation_and_bucket_parsing_are_deterministic() -> None:
    trace = generate_dynamic_trace(
        steps=3,
        max_batch_size=4,
        min_seq_len=2,
        max_seq_len=8,
        seed=7,
        prefill_interval=2,
        mixed_interval=3,
    )

    assert parse_batch_buckets("1, 2", max_batch_size=4) == (1, 2, 4)
    assert len(trace) == 3
    assert [step.phase for step in trace] == ["decode", "prefill", "mixed"]
    assert all(1 <= step.active_batch_size <= 4 for step in trace)


def test_representative_trace_steps_cover_first_seen_buckets() -> None:
    trace = (
        TraceStep(active_batch_size=1, seq_len=2, phase="decode", queue_wait_ms=0.0),
        TraceStep(active_batch_size=3, seq_len=4, phase="decode", queue_wait_ms=0.0),
        TraceStep(active_batch_size=2, seq_len=3, phase="mixed", queue_wait_ms=0.0),
        TraceStep(active_batch_size=5, seq_len=5, phase="prefill", queue_wait_ms=0.0),
    )

    selected = representative_trace_steps(trace, batch_buckets=(1, 2, 4, 8))

    assert selected == (
        (0, trace[0]),
        (2, trace[2]),
        (1, trace[1]),
        (3, trace[3]),
    )


def test_dynamic_trace_accounting_averages_variable_steps() -> None:
    trace = (
        TraceStep(active_batch_size=1, seq_len=2, phase="decode", queue_wait_ms=0.0),
        TraceStep(active_batch_size=2, seq_len=3, phase="decode", queue_wait_ms=0.0),
    )

    assert average_trace_memory_traffic_bytes(
        trace,
        hidden_dim=4,
        intermediate_dim=8,
        num_heads=2,
        head_dim=2,
        dtype_size=2,
    ) == 496
    assert average_trace_flop_count(
        trace,
        hidden_dim=4,
        intermediate_dim=8,
        num_heads=2,
        head_dim=2,
    ) == 444


def test_decode_step_memory_traffic_estimate() -> None:
    assert (
        decode_step_memory_traffic_bytes(
            batch_size=1,
            hidden_dim=4,
            intermediate_dim=8,
            seq_len=2,
            num_heads=2,
            head_dim=2,
            dtype_size=2,
        )
        == 376
    )


def test_decode_step_flop_estimate() -> None:
    assert (
        decode_step_flop_count(
            batch_size=1,
            hidden_dim=4,
            intermediate_dim=8,
            seq_len=2,
            num_heads=2,
            head_dim=2,
        )
        == 279
    )


def test_decode_step_models_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        decode_step_memory_traffic_bytes(
            batch_size=0,
            hidden_dim=4,
            intermediate_dim=8,
            seq_len=2,
            num_heads=2,
            head_dim=2,
            dtype_size=2,
        )

    with pytest.raises(ValueError, match="intermediate_dim"):
        decode_step_flop_count(
            batch_size=1,
            hidden_dim=4,
            intermediate_dim=3,
            seq_len=2,
            num_heads=2,
            head_dim=2,
        )


@requires_torch
def test_decode_step_naive_eager_records_extended_metrics() -> None:
    result = run_one(
        torch=torch,
        inputs=_tiny_inputs(),
        mode="naive-eager",
        dtype=torch.float32,
        device="cpu",
        eps=1e-6,
        warmup=0,
        iterations=1,
        reference=None,
    )

    assert result.name == "naive:decode_step"
    assert result.strategy == "naive-eager"
    assert result.metrics is not None
    assert result.metrics["tokens_per_step"] == 1
    assert result.metrics["tokens_per_second_p50"] > 0


def _tiny_inputs() -> object:
    from cuda_kernel_lab.benchmarks.decode_step import DecodeStepInputs

    return DecodeStepInputs(
        x=torch.randn((1, 4), dtype=torch.float32),
        rms_weight=torch.randn((4,), dtype=torch.float32),
        q_weight=torch.randn((4, 4), dtype=torch.float32) * 0.02,
        gate_weight=torch.randn((4, 4), dtype=torch.float32) * 0.02,
        up_weight=torch.randn((4, 4), dtype=torch.float32) * 0.02,
        key_cache=torch.randn((1, 2, 2, 2), dtype=torch.float32),
        value_cache=torch.randn((1, 2, 2, 2), dtype=torch.float32),
    )

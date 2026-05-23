from __future__ import annotations

import json
from pathlib import Path

import pytest

from cuda_kernel_lab import benchmark_report


def test_load_report_rows_computes_speedups_and_noise(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "memory.jsonl",
        [
            _record(
                benchmark="memory_bandwidth",
                name="torch:vector_add",
                dtype="float32",
                p50=2.0,
                p95=2.1,
                p99=2.2,
                gbps=50.0,
                args={"block_size": 1024, "warmup": 25, "iterations": 100},
            ),
            _record(
                benchmark="memory_bandwidth",
                name="triton:vector_add",
                dtype="float32",
                p50=1.0,
                p95=1.4,
                p99=1.5,
                gbps=100.0,
                args={"block_size": 1024, "warmup": 25, "iterations": 100},
            ),
        ],
    )

    rows = benchmark_report.load_report_rows(tmp_path)
    triton = next(row for row in rows if row.backend == "triton")

    assert triton.primitive == "memory"
    assert triton.operation == "vector_add"
    assert triton.variant == "block_size=1024"
    assert triton.optimization.method_family == "launch tuning"
    assert triton.optimization.technique == "Coalesced block-size tuning"
    assert triton.speedup_vs_torch == pytest.approx(2.0)
    assert triton.noise_ratio == pytest.approx(1.4)


def test_load_report_rows_reads_all_jsonl_files(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "vector-add-block-size.jsonl",
        [
            _record(
                benchmark="memory_bandwidth",
                name="triton:vector_add",
                dtype="float32",
                p50=1.0,
                p95=1.1,
                p99=1.2,
                gbps=100.0,
                args={"block_size": 2048, "warmup": 25, "iterations": 100},
            ),
        ],
    )

    rows = benchmark_report.load_report_rows(tmp_path)

    assert [row.source.name for row in rows] == ["vector-add-block-size.jsonl"]
    assert rows[0].variant == "block_size=2048"


def test_render_markdown_includes_fastest_and_backend_detail(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "softmax.jsonl",
        [
            _record(
                benchmark="softmax",
                name="torch:softmax",
                dtype="float16",
                shape=(4096, 1024),
                p50=3.0,
                p95=3.1,
                p99=3.2,
                gbps=25.0,
                tflops=0.5,
            ),
            _record(
                benchmark="softmax",
                name="triton:softmax",
                dtype="float16",
                shape=(4096, 1024),
                p50=1.5,
                p95=1.6,
                p99=1.7,
                gbps=50.0,
                tflops=1.0,
            ),
        ],
    )

    rows = benchmark_report.load_report_rows(tmp_path)
    report = benchmark_report.render_markdown(rows, input_dir=tmp_path)

    assert "# GPU Benchmark Report" in report
    assert "Status: generated from benchmark JSONL" in report
    assert "- Git commit: `abc123`" in report
    assert "- CUDA devices: `NVIDIA A10G" in report
    assert "## Optimization Techniques Tested" in report
    assert "| fusion | Row-wise softmax fusion | softmax softmax |" in report
    assert (
        "| softmax | softmax | float16 | 4096x1024 | "
        "traffic_model=fused | triton | Row-wise softmax fusion | 1.5 | 50 | 1 |"
    ) in report
    assert "- Loaded 2 benchmark rows from 1 result file." in report
    assert (
        "- Largest Triton wins vs torch: softmax softmax float16 traffic_model=fused (2x)."
        in report
    )
    assert (
        "- Fusion techniques produced the strongest Triton wins by removing "
        "intermediate traffic or launch overhead: softmax softmax float16 "
        "traffic_model=fused (2x)."
    ) in report
    assert "| softmax | softmax | float16 | 4096x1024 | traffic_model=fused" in report
    assert (
        "| triton | triton-fused-row-softmax | Row-wise softmax fusion | pass | "
        "1.5 | 1.6 | 1.7 |"
    ) in report
    assert "| 2 | 1.067 |" in report


def test_render_markdown_includes_dynamic_trace_detail(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "decode-step-dynamic-tail.jsonl",
        [
            _record(
                benchmark="decode_step",
                name="dynamic-piecewise-graph-same-stream:decode_step",
                dtype="float16",
                shape=(8, 2048, 16, 64, 4096),
                p50=0.45,
                p95=1.1,
                p99=1.3,
                gbps=80.0,
                tflops=0.2,
                args={
                    "batch_buckets": "1,2,4,8",
                    "mode": "dynamic-piecewise-graph-same-stream",
                    "seed": 7,
                    "warmup": 25,
                    "iterations": 500,
                },
                metrics={
                    "tokens_per_second": 7600.0,
                    "tokens_per_second_at_host_p95": 4100.0,
                    "scheduler_cpu_p95_us": 1100.0,
                    "host_tail_ratio_p95_p50": 2.44,
                    "padding_waste_pct": 9.5,
                    "bucket_breakdown": {
                        "1": {
                            "steps": 10,
                            "host_p50_ms": 0.4,
                            "host_p95_ms": 0.45,
                            "host_p99_ms": 0.48,
                            "host_tail_ratio_p95_p50": 1.12,
                            "padding_waste_pct": 0.0,
                        },
                        "8": {
                            "steps": 30,
                            "host_p50_ms": 0.8,
                            "host_p95_ms": 1.25,
                            "host_p99_ms": 1.35,
                            "host_tail_ratio_p95_p50": 1.56,
                            "padding_waste_pct": 12.5,
                        },
                    },
                    "orchestration_breakdown": {
                        "input_copy_host_ms": {
                            "samples": 500,
                            "host_p50_ms": 0.06,
                            "host_p95_ms": 0.08,
                            "host_p99_ms": 0.1,
                            "total_host_ms": 32.0,
                        }
                    },
                },
            )
        ],
    )

    rows = benchmark_report.load_report_rows(tmp_path)
    report = benchmark_report.render_markdown(rows, input_dir=tmp_path)

    assert "## Dynamic Trace Detail" in report
    assert "### Tail Policy Summary" in report
    assert "| 1,2,4,8 | 1 | 0.45 | 1.1 | 1.3 | 7600 | 4100 | 9.5 | 1.25 |" in report
    assert "### Tail Sweep" in report
    assert "| dynamic-piecewise-graph-same-stream | 1,2,4,8 | 7 | 0.45 | 1.1 |" in report
    assert "8 (p95 1.25 ms)" in report
    assert "### Worst Dynamic Buckets" in report
    assert "| dynamic-piecewise-graph-same-stream | `decode-step-dynamic-tail.jsonl` |" in report
    assert "### Host Orchestration" in report
    assert "| dynamic-piecewise-graph-same-stream | 1,2,4,8 | 7 | input_copy_host_ms |" in report


def test_render_markdown_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no benchmark JSONL records"):
        benchmark_report.render_markdown([], input_dir=tmp_path)


def test_report_dry_run_prints_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "report.md"
    _write_jsonl(
        tmp_path / "norms.jsonl",
        [
            _record(
                benchmark="norms",
                name="torch:rmsnorm",
                dtype="float32",
                shape=(4096, 4096),
                p50=4.0,
                p95=4.2,
                p99=4.3,
                gbps=75.0,
            )
        ],
    )

    benchmark_report.main(
        [
            "--input-dir",
            str(tmp_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )

    assert "# GPU Benchmark Report" in capsys.readouterr().out
    assert not output_path.exists()


def test_default_output_uses_run_id_layout() -> None:
    assert benchmark_report.default_output_for(
        Path("experiments/results/runpod/2026-05-19-l4-baseline")
    ) == Path("experiments/reports/runpod/2026-05-19-l4-baseline.md")


def test_default_output_preserves_legacy_aws_layout() -> None:
    assert benchmark_report.default_output_for(
        Path("experiments/results/aws-ec2/2026-05-19-a10g-baseline")
    ) == Path("experiments/reports/aws-ec2/2026-05-19-a10g-baseline.md")


def test_default_output_falls_back_to_reports_dir() -> None:
    assert benchmark_report.default_output_for(Path("custom/results")) == Path(
        "experiments/reports/runpod/benchmark-report.md"
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def _record(
    *,
    benchmark: str,
    name: str,
    dtype: str,
    p50: float,
    p95: float,
    p99: float,
    gbps: float,
    shape: tuple[int, ...] = (16_777_216,),
    tflops: float = 0.0,
    args: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run": {
            "benchmark": benchmark,
            "args": _args_for(benchmark) if args is None else args,
            "command": "uv run benchmark",
            "timestamp_utc": "2026-05-19T00:00:00+00:00",
            "git_commit": "abc123",
            "git_dirty": False,
            "host": {"python": "3.13.2", "platform": "Linux"},
            "packages": {"torch": "2.3.0", "triton": "2.3.0"},
            "cuda_devices": [
                {
                    "index": 0,
                    "name": "NVIDIA A10G",
                    "capability": [8, 6],
                    "total_memory_bytes": 24_000_000_000,
                    "multiprocessor_count": 80,
                }
            ],
        },
        "result": {
            "name": name,
            "device": "cuda",
            "dtype": dtype,
            "shape": list(shape),
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "bytes_moved": 1,
            "bandwidth_gbps": gbps,
            "flops": 1,
            "tflops": tflops,
            "latencies_ms": [p50, p95, p99],
            "strategy": _strategy_for(name),
            "variant": _variant_for(benchmark, args or _args_for(benchmark)),
            "parameters": args or _args_for(benchmark),
            "metrics": metrics or {},
            "correctness": {
                "checked": True,
                "passed": True,
                "reference_backend": "torch",
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": 1e-5,
                "rtol": 1e-4,
                "message": None,
            },
        },
    }


def _args_for(benchmark: str) -> dict[str, object]:
    if benchmark == "memory_bandwidth":
        return {"block_size": 1024, "warmup": 25, "iterations": 100}
    if benchmark == "softmax":
        return {"traffic_model": "fused", "warmup": 25, "iterations": 100}
    if benchmark == "norms":
        return {"eps": None, "warmup": 25, "iterations": 100}
    return {}


def _variant_for(benchmark: str, args: dict[str, object]) -> str:
    if benchmark == "memory_bandwidth":
        return f"block_size={args['block_size']}"
    if benchmark == "softmax":
        return "traffic_model=fused"
    if benchmark == "norms":
        return "eps=1e-06"
    return "default"


def _strategy_for(name: str) -> str:
    backend, operation = name.split(":", maxsplit=1)
    if backend == "torch":
        return "torch-baseline"
    if operation == "decode_step":
        return backend
    if operation == "softmax":
        return "triton-fused-row-softmax"
    return f"triton-{operation}"

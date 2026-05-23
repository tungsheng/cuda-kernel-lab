"""Generate Markdown reports from benchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda_kernel_lab.optimization import (
    OptimizationTechnique,
    technique_from_mapping,
    technique_from_result,
)
from cuda_kernel_lab.roofline import (
    RooflineSpec,
    arithmetic_intensity,
    ridge_point_flops_per_byte,
    spec_for_device_name,
)

DEFAULT_INPUT_DIR = Path("experiments/results/runpod/manual-run")
RUN_RESULTS_ROOT = Path("experiments/results/runpod")
RUN_REPORT_ROOT = Path("experiments/reports/runpod")
LEGACY_AWS_RESULTS_ROOT = Path("experiments/results/aws-ec2")
LEGACY_AWS_REPORT_ROOT = Path("experiments/reports/aws-ec2")
DEFAULT_OUTPUT = RUN_REPORT_ROOT / "benchmark-report.md"
NOISE_RATIO_THRESHOLD = 1.20


@dataclass(frozen=True)
class ReportRow:
    primitive: str
    operation: str
    backend: str
    strategy: str
    optimization: OptimizationTechnique
    dtype: str
    shape: tuple[int, ...]
    variant: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    bytes_moved: int
    bandwidth_gbps: float
    flops: int
    tflops: float
    speedup_vs_torch: float | None
    noise_ratio: float | None
    correctness: str
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    run: dict[str, Any]
    source: Path


def load_report_rows(input_dir: Path) -> list[ReportRow]:
    """Load all benchmark rows from JSONL files in a benchmark run directory."""

    rows: list[ReportRow] = []
    for path in _jsonl_paths(input_dir):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(
                    _row_from_record(
                        json.loads(stripped),
                        source=path,
                        line_number=line_number,
                    )
                )

    return _with_speedups(rows)


def render_markdown(rows: list[ReportRow], *, input_dir: Path) -> str:
    """Render a compact Markdown report."""

    if not rows:
        raise ValueError(f"no benchmark JSONL records found under {input_dir}")

    lines = [
        "# GPU Benchmark Report",
        "",
        "Status: generated from benchmark JSONL",
        "",
        "## Question",
        "",
        "What are the baseline PyTorch and Triton measurements for this CUDA",
        "Kernel Lab benchmark run?",
        "",
        "## Result Files",
        "",
    ]
    for source in sorted({row.source for row in rows}):
        lines.append(f"- `{source}`")

    first_run = rows[0].run
    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Git commit: `{first_run.get('git_commit') or ''}`",
            f"- Git dirty: `{first_run.get('git_dirty')}`",
            *_provider_lines(first_run.get("provider")),
            f"- Python: `{_nested(first_run, 'host', 'python')}`",
            f"- Platform: `{_nested(first_run, 'host', 'platform')}`",
            f"- PyTorch: `{_nested(first_run, 'packages', 'torch')}`",
            f"- Triton: `{_nested(first_run, 'packages', 'triton')}`",
            f"- CUDA devices: {_cuda_devices_label(first_run.get('cuda_devices'))}",
        ]
    )

    lines.extend(
        [
            "",
            "## Optimization Techniques Tested",
            "",
            "| Family | Technique | Used By | Hypothesis |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(_optimization_technique_rows(rows))

    lines.extend(
        [
            "",
            "## Fastest By Operation",
            "",
            "| Primitive | Operation | Dtype | Shape | Variant | Fastest Backend | "
            "Technique | p50 ms | GB/s | TFLOP/s |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _fastest_rows(rows):
        lines.append(
            "| "
            f"{row.primitive} | {row.operation} | {row.dtype} | {_shape_label(row.shape)} | "
            f"{row.variant} | {row.backend} | {row.optimization.technique} | "
            f"{_fmt(row.p50_ms)} | "
            f"{_fmt(row.bandwidth_gbps)} | {_fmt(row.tflops)} |"
        )

    roofline_lines = _roofline_summary_lines(rows)
    if roofline_lines:
        lines.extend(
            [
                "",
                "## Roofline Summary",
                "",
                *roofline_lines,
            ]
        )

    lines.extend(
        [
            "",
            "## Backend Detail",
            "",
            "| Primitive | Operation | Dtype | Shape | Variant | Backend | Strategy | Technique | "
            "Correct | p50 ms | p95 ms | p99 ms | GB/s | TFLOP/s | Speedup vs Torch | Noise |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | "
            "---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=_sort_key):
        lines.append(
            "| "
            f"{row.primitive} | {row.operation} | {row.dtype} | {_shape_label(row.shape)} | "
            f"{row.variant} | {row.backend} | {row.strategy} | "
            f"{row.optimization.technique} | {row.correctness} | "
            f"{_fmt(row.p50_ms)} | {_fmt(row.p95_ms)} | {_fmt(row.p99_ms)} | "
            f"{_fmt(row.bandwidth_gbps)} | {_fmt(row.tflops)} | "
            f"{_fmt_optional(row.speedup_vs_torch)} | "
            f"{_noise_label(row.noise_ratio)} |"
        )

    dynamic_detail_lines = _dynamic_detail_lines(rows)
    if dynamic_detail_lines:
        lines.extend(
            [
                "",
                "## Dynamic Trace Detail",
                "",
                *dynamic_detail_lines,
            ]
        )

    lines.extend(
        [
            "",
            "## Observation",
            "",
            *_observation_lines(rows),
            "",
            "## Technique Takeaways",
            "",
            *_technique_takeaway_lines(rows),
            "",
            "## Interpretation",
            "",
            *_interpretation_lines(rows),
            "",
            "## Next Question",
            "",
            _next_question(rows),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print report without writing it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_report_rows(args.input_dir)
    report = render_markdown(rows, input_dir=args.input_dir)
    output = args.output or default_output_for(args.input_dir)
    if args.dry_run:
        print(report)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote benchmark report to {output}", file=sys.stderr)


def default_output_for(input_dir: Path) -> Path:
    """Return the report path that matches a benchmark input directory."""

    normalized = input_dir
    if normalized.is_absolute():
        try:
            normalized = normalized.relative_to(Path.cwd())
        except ValueError:
            return DEFAULT_OUTPUT

    try:
        run_path = normalized.relative_to(RUN_RESULTS_ROOT)
    except ValueError:
        try:
            run_path = normalized.relative_to(LEGACY_AWS_RESULTS_ROOT)
        except ValueError:
            return DEFAULT_OUTPUT
        if len(run_path.parts) != 1 or not run_path.parts[0]:
            return DEFAULT_OUTPUT
        return LEGACY_AWS_REPORT_ROOT / f"{run_path.parts[0]}.md"

    if len(run_path.parts) != 1 or not run_path.parts[0]:
        return DEFAULT_OUTPUT
    return RUN_REPORT_ROOT / f"{run_path.parts[0]}.md"


def _provider_lines(provider: object) -> list[str]:
    if not isinstance(provider, dict) or not provider:
        return []

    name = provider.get("name") or provider.get("platform")
    fields = [f"- Provider: `{name or 'unknown'}`"]
    provider_id = provider.get("id") or provider.get("pod_id") or provider.get("instance_id")
    if provider_id:
        fields.append(f"- Provider id: `{provider_id}`")
    gpu_id = provider.get("gpu_id")
    if gpu_id:
        fields.append(f"- Provider GPU: `{gpu_id}`")
    cloud_type = provider.get("cloud_type")
    if cloud_type:
        fields.append(f"- Provider cloud: `{cloud_type}`")
    image = provider.get("image")
    if image:
        fields.append(f"- Provider image: `{image}`")
    return fields


def _row_from_record(record: dict[str, Any], *, source: Path, line_number: int) -> ReportRow:
    try:
        run = record["run"]
        result = record["result"]
        backend, operation = str(result["name"]).split(":", maxsplit=1)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid benchmark record in {source}:{line_number}") from exc

    primitive = _primitive_label(str(run["benchmark"]))
    parameters = result.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    strategy = str(result.get("strategy") or _strategy_label(backend))
    optimization = technique_from_mapping(result.get("optimization")) or technique_from_result(
        backend=backend,
        primitive=primitive,
        operation=operation,
        strategy=strategy,
        parameters=parameters,
    )

    return ReportRow(
        primitive=primitive,
        operation=operation,
        backend=backend,
        strategy=strategy,
        optimization=optimization,
        dtype=str(result["dtype"]),
        shape=tuple(int(dim) for dim in result["shape"]),
        variant=str(result.get("variant") or _variant_label(run)),
        p50_ms=float(result["p50_ms"]),
        p95_ms=float(result["p95_ms"]),
        p99_ms=float(result["p99_ms"]),
        bytes_moved=int(result.get("bytes_moved") or 0),
        bandwidth_gbps=float(result["bandwidth_gbps"]),
        flops=int(result.get("flops") or 0),
        tflops=float(result["tflops"]),
        speedup_vs_torch=None,
        noise_ratio=_ratio(float(result["p95_ms"]), float(result["p50_ms"])),
        correctness=_correctness_label(result.get("correctness")),
        parameters=parameters,
        metrics=metrics,
        run=run,
        source=source,
    )


def _jsonl_paths(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())


def _with_speedups(rows: list[ReportRow]) -> list[ReportRow]:
    torch_p50 = {
        (row.primitive, row.operation, row.dtype, row.shape, row.variant): row.p50_ms
        for row in rows
        if row.backend == "torch"
    }
    enriched = []
    for row in rows:
        baseline = torch_p50.get((row.primitive, row.operation, row.dtype, row.shape, row.variant))
        speedup = _ratio(baseline, row.p50_ms) if baseline is not None else None
        enriched.append(
            ReportRow(
                primitive=row.primitive,
                operation=row.operation,
                backend=row.backend,
                strategy=row.strategy,
                optimization=row.optimization,
                dtype=row.dtype,
                shape=row.shape,
                variant=row.variant,
                p50_ms=row.p50_ms,
                p95_ms=row.p95_ms,
                p99_ms=row.p99_ms,
                bytes_moved=row.bytes_moved,
                bandwidth_gbps=row.bandwidth_gbps,
                flops=row.flops,
                tflops=row.tflops,
                speedup_vs_torch=speedup,
                noise_ratio=row.noise_ratio,
                correctness=row.correctness,
                parameters=row.parameters,
                metrics=row.metrics,
                run=row.run,
                source=row.source,
            )
        )
    return enriched


def _fastest_rows(rows: list[ReportRow]) -> list[ReportRow]:
    fastest: dict[tuple[str, str, str, tuple[int, ...], str], ReportRow] = {}
    for row in rows:
        key = (row.primitive, row.operation, row.dtype, row.shape, row.variant)
        current = fastest.get(key)
        if current is None or row.p50_ms < current.p50_ms:
            fastest[key] = row
    return sorted(fastest.values(), key=_sort_key)


def _optimization_technique_rows(rows: list[ReportRow]) -> list[str]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        optimization = row.optimization
        summary = summaries.setdefault(
            optimization.method_id,
            {
                "family": optimization.method_family,
                "technique": optimization.technique,
                "hypothesis": optimization.hypothesis or optimization.description,
                "used_by": set(),
            },
        )
        summary["used_by"].add(_technique_use_label(row))

    lines = []
    for _method_id, summary in sorted(
        summaries.items(),
        key=lambda item: _method_sort_key(str(item[1]["family"]), item[0]),
    ):
        used_by = _used_by_label(sorted(summary["used_by"]))
        lines.append(
            "| "
            f"{summary['family']} | {summary['technique']} | {used_by} | "
            f"{_escape_cell(str(summary['hypothesis']))} |"
        )
    return lines


def _roofline_summary_lines(rows: list[ReportRow]) -> list[str]:
    spec = _roofline_spec_for_rows(rows)
    if spec is None:
        return []

    candidates = [
        row
        for row in _fastest_rows(rows)
        if row.bytes_moved > 0 and (row.flops > 0 or row.bandwidth_gbps > 0)
    ]
    if not candidates:
        return []

    lines = [
        f"- Spec: `{spec.name}` ({spec.source}).",
        f"- Peak HBM bandwidth: `{_fmt(spec.memory_bandwidth_gbps)} GB/s`.",
        "",
        "| Primitive | Operation | Dtype | Shape | Backend | Strategy | "
        "Intensity FLOP/B | Achieved GB/s | HBM Peak % | Achieved TFLOP/s | "
        "Math Peak % | Bound |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(candidates, key=lambda row: _roofline_sort_key(row, spec))[:12]:
        lines.append(
            "| "
            f"{row.primitive} | {row.operation} | {row.dtype} | {_shape_label(row.shape)} | "
            f"{row.backend} | {row.strategy} | "
            f"{_fmt_optional(arithmetic_intensity(row.flops, row.bytes_moved))} | "
            f"{_fmt(row.bandwidth_gbps)} | "
            f"{_fmt_optional(_memory_peak_pct(row, spec))} | "
            f"{_fmt(row.tflops)} | "
            f"{_fmt_optional(_math_peak_pct(row, spec))} | "
            f"{_roofline_bound_label(row, spec)} |"
        )
    return lines


def _roofline_spec_for_rows(rows: list[ReportRow]) -> RooflineSpec | None:
    first_run = rows[0].run
    provider = first_run.get("provider")
    if isinstance(provider, dict):
        spec = spec_for_device_name(str(provider.get("gpu_id") or ""))
        if spec is not None:
            return spec

    cuda_devices = first_run.get("cuda_devices")
    if isinstance(cuda_devices, list):
        for device in cuda_devices:
            if isinstance(device, dict):
                spec = spec_for_device_name(str(device.get("name") or ""))
                if spec is not None:
                    return spec
    return None


def _roofline_sort_key(row: ReportRow, spec: RooflineSpec) -> tuple[int, float]:
    peak_pct = max(_math_peak_pct(row, spec) or 0.0, _memory_peak_pct(row, spec) or 0.0)
    return (0 if row.primitive == "matmul" else 1, -peak_pct)


def _memory_peak_pct(row: ReportRow, spec: RooflineSpec) -> float | None:
    if spec.memory_bandwidth_gbps <= 0:
        return None
    return row.bandwidth_gbps / spec.memory_bandwidth_gbps * 100.0


def _math_peak_pct(row: ReportRow, spec: RooflineSpec) -> float | None:
    peak_tflops = spec.peak_tflops_for_dtype(row.dtype)
    if peak_tflops is None or peak_tflops <= 0:
        return None
    return row.tflops / peak_tflops * 100.0


def _roofline_bound_label(row: ReportRow, spec: RooflineSpec) -> str:
    if row.flops <= 0:
        return "memory"
    intensity = arithmetic_intensity(row.flops, row.bytes_moved)
    ridge_point = ridge_point_flops_per_byte(spec, row.dtype)
    if intensity is None or ridge_point is None:
        return ""
    return "compute" if intensity >= ridge_point else "memory"


def _technique_takeaway_lines(rows: list[ReportRow]) -> list[str]:
    lines: list[str] = []
    families = {row.optimization.method_family for row in rows}

    if "fusion" in families:
        fusion_wins = [
            row
            for row in rows
            if row.backend == "triton"
            and row.optimization.method_family == "fusion"
            and row.speedup_vs_torch is not None
            and row.speedup_vs_torch > 1
        ]
        if fusion_wins:
            top = sorted(
                fusion_wins,
                key=lambda row: row.speedup_vs_torch or 0,
                reverse=True,
            )[:3]
            lines.append(
                "- Fusion techniques produced the strongest Triton wins by removing "
                f"intermediate traffic or launch overhead: {_row_list_label(top)}."
            )
        else:
            lines.append(
                "- Fusion rows should be read as tests of intermediate-traffic removal, "
                "not as generic Triton-vs-PyTorch comparisons."
            )

    if "launch tuning" in families:
        launch_rows = [
            row
            for row in rows
            if row.backend == "triton" and row.optimization.method_family == "launch tuning"
        ]
        if launch_rows and all((row.speedup_vs_torch or 0) <= 1 for row in launch_rows):
            lines.append(
                "- Launch tuning for simple coalesced memory kernels did not beat PyTorch; "
                "compare GB/s and profiler DRAM throughput before adding wider block-size sweeps."
            )
        elif launch_rows:
            lines.append(
                "- Launch-tuning rows isolate block-size effects, so compare rows with the "
                "same shape, dtype, and traffic model."
            )

    if "reduction" in families:
        lines.append(
            "- Reduction-strategy rows separate first-pass streaming bandwidth from "
            "end-to-end launch and finalization cost."
        )

    if "tiling" in families:
        tiling_rows = [
            row
            for row in rows
            if row.backend == "triton" and row.optimization.method_family == "tiling"
        ]
        if tiling_rows:
            best = max(tiling_rows, key=lambda row: row.tflops)
            lines.append(
                "- Tiled matmul rows should be judged by TFLOP/s and Tensor Core counters; "
                f"the current best Triton tile/launch config is {_row_label(best)} at "
                f"{_fmt(best.tflops)} TFLOP/s."
            )
        else:
            lines.append(
                "- Tiling rows should be paired with Tensor Core, register, and occupancy "
                "counters before choosing a final tile and launch configuration."
            )

    return lines or ["- No cataloged optimization techniques were found in this result set."]


def _dynamic_detail_lines(rows: list[ReportRow]) -> list[str]:
    dynamic_rows = [
        row
        for row in rows
        if row.primitive == "decode_step" and isinstance(row.metrics.get("bucket_breakdown"), dict)
    ]
    if not dynamic_rows:
        return []

    lines: list[str] = []
    tail_rows = _dynamic_tail_rows(dynamic_rows)
    if tail_rows:
        lines.extend(_tail_policy_summary_lines(tail_rows))
        lines.append("")
        lines.extend(
            [
                "### Tail Sweep",
                "",
                "| Strategy | Buckets | Seed | p50 ms | p95 ms | p99 ms | p95/p50 | "
                "tok/s | tok/s @ p95 | scheduler p95 us | Pad % | Worst Bucket |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | --- |",
            ]
        )
        for row in sorted(tail_rows, key=_dynamic_row_sort_key):
            worst_bucket = _worst_bucket(row)
            lines.append(
                "| "
                f"{_escape_cell(row.strategy)} | {_escape_cell(_bucket_policy_label(row))} | "
                f"{_seed_label(row)} | {_fmt(row.p50_ms)} | {_fmt(row.p95_ms)} | "
                f"{_fmt(row.p99_ms)} | "
                f"{_fmt_metric(row.metrics, 'host_tail_ratio_p95_p50')} | "
                f"{_fmt_metric(row.metrics, 'tokens_per_second')} | "
                f"{_fmt_metric(row.metrics, 'tokens_per_second_at_host_p95')} | "
                f"{_fmt_optional(_scheduler_p95_us(row.metrics))} | "
                f"{_fmt_metric(row.metrics, 'padding_waste_pct')} | "
                f"{_escape_cell(_worst_bucket_label(worst_bucket))} |"
            )

    worst_bucket_rows = _worst_bucket_rows(dynamic_rows)
    if worst_bucket_rows:
        if lines:
            lines.append("")
        lines.extend(
            [
                "### Worst Dynamic Buckets",
                "",
                "| Strategy | Source | Buckets | Seed | Bucket | Steps | p50 ms | p95 ms | "
                "p99 ms | p95/p50 | Pad % |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row, bucket, metrics in worst_bucket_rows[:10]:
            lines.append(
                "| "
                f"{_escape_cell(row.strategy)} | `{row.source.name}` | "
                f"{_escape_cell(_bucket_policy_label(row))} | {_seed_label(row)} | "
                f"{bucket} | {_fmt_int_metric(metrics, 'steps')} | "
                f"{_fmt_metric(metrics, 'host_p50_ms')} | "
                f"{_fmt_metric(metrics, 'host_p95_ms')} | "
                f"{_fmt_metric(metrics, 'host_p99_ms')} | "
                f"{_fmt_metric(metrics, 'host_tail_ratio_p95_p50')} | "
                f"{_fmt_metric(metrics, 'padding_waste_pct')} |"
            )

    orchestration_rows = _orchestration_rows(tail_rows or dynamic_rows)
    if orchestration_rows:
        if lines:
            lines.append("")
        lines.extend(
            [
                "### Host Orchestration",
                "",
                "| Strategy | Buckets | Seed | Region | Samples | p50 ms | p95 ms | "
                "p99 ms | Total ms |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row, region, metrics in orchestration_rows[:40]:
            lines.append(
                "| "
                f"{_escape_cell(row.strategy)} | {_escape_cell(_bucket_policy_label(row))} | "
                f"{_seed_label(row)} | {_escape_cell(region)} | "
                f"{_fmt_int_metric(metrics, 'samples')} | "
                f"{_fmt_metric(metrics, 'host_p50_ms')} | "
                f"{_fmt_metric(metrics, 'host_p95_ms')} | "
                f"{_fmt_metric(metrics, 'host_p99_ms')} | "
                f"{_fmt_metric(metrics, 'total_host_ms')} |"
            )

    return lines


def _tail_policy_summary_lines(rows: list[ReportRow]) -> list[str]:
    groups: dict[str, list[ReportRow]] = {}
    for row in rows:
        groups.setdefault(_bucket_policy_label(row), []).append(row)

    summary_rows = []
    for bucket_policy, bucket_rows in groups.items():
        worst_bucket_p95 = []
        for row in bucket_rows:
            worst_bucket = _worst_bucket(row)
            if worst_bucket is None:
                continue
            _bucket, worst_metrics = worst_bucket
            worst_bucket_p95.append(_numeric_metric(worst_metrics, "host_p95_ms"))
        summary_rows.append(
            (
                bucket_policy,
                len(bucket_rows),
                _mean(row.p50_ms for row in bucket_rows),
                _mean(row.p95_ms for row in bucket_rows),
                _mean(row.p99_ms for row in bucket_rows),
                _mean(
                    value
                    for row in bucket_rows
                    if (value := _numeric_metric(row.metrics, "tokens_per_second")) is not None
                ),
                _mean(
                    value
                    for row in bucket_rows
                    if (
                        value := _numeric_metric(
                            row.metrics,
                            "tokens_per_second_at_host_p95",
                        )
                    )
                    is not None
                ),
                _mean(
                    value
                    for row in bucket_rows
                    if (value := _numeric_metric(row.metrics, "padding_waste_pct")) is not None
                ),
                _mean(value for value in worst_bucket_p95 if value is not None),
            )
        )

    lines = [
        "### Tail Policy Summary",
        "",
        "| Buckets | Runs | Avg p50 ms | Avg p95 ms | Avg p99 ms | Avg tok/s | "
        "Avg tok/s @ p95 | Avg Pad % | Avg Worst Bucket p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (
        bucket_policy,
        run_count,
        p50_ms,
        p95_ms,
        p99_ms,
        tokens_per_second,
        tokens_per_second_at_p95,
        padding_waste_pct,
        worst_bucket_p95_ms,
    ) in sorted(summary_rows, key=lambda row: row[3]):
        lines.append(
            "| "
            f"{_escape_cell(bucket_policy)} | {run_count} | {_fmt_optional(p50_ms)} | "
            f"{_fmt_optional(p95_ms)} | {_fmt_optional(p99_ms)} | "
            f"{_fmt_optional(tokens_per_second)} | "
            f"{_fmt_optional(tokens_per_second_at_p95)} | "
            f"{_fmt_optional(padding_waste_pct)} | {_fmt_optional(worst_bucket_p95_ms)} |"
        )
    return lines


def _observation_lines(rows: list[ReportRow]) -> list[str]:
    sources = {row.source for row in rows}
    correctness = Counter(row.correctness for row in rows)
    fastest_backends = Counter(row.backend for row in _fastest_rows(rows))
    noisy_rows = _noisy_rows(rows)
    triton_wins = _top_triton_wins(rows)

    lines = [
        f"- Loaded {len(rows)} {_plural(len(rows), 'benchmark row')} from "
        f"{len(sources)} {_plural(len(sources), 'result file')}.",
        f"- Fastest backend split: {_counter_label(fastest_backends)}.",
    ]
    if correctness == Counter({"pass": len(rows)}):
        lines.append(f"- All {len(rows)} correctness checks passed.")
    else:
        lines.append(f"- Correctness summary: {_counter_label(correctness)}.")

    if triton_wins:
        lines.append(f"- Largest Triton wins vs torch: {_row_list_label(triton_wins)}.")
    else:
        lines.append("- No Triton rows beat the matching torch baseline in this result set.")

    if noisy_rows:
        lines.append(
            f"- Noisy rows at p95/p50 >= {_fmt(NOISE_RATIO_THRESHOLD)}: "
            f"{_row_list_label(noisy_rows)}."
        )
    else:
        threshold = _fmt(NOISE_RATIO_THRESHOLD)
        lines.append(f"- No rows exceeded the {threshold} p95/p50 noise threshold.")

    return lines


def _interpretation_lines(rows: list[ReportRow]) -> list[str]:
    lines: list[str] = []
    fastest = _fastest_rows(rows)

    if any(
        row.backend == "triton" and row.speedup_vs_torch is not None and row.speedup_vs_torch >= 2
        for row in rows
    ):
        lines.append(
            "- Triton is strongest where a fused kernel removes framework overhead or "
            "intermediate memory traffic."
        )

    memory_fastest = [row for row in fastest if row.primitive == "memory"]
    if memory_fastest and all(row.backend == "torch" for row in memory_fastest):
        lines.append(
            "- Memory primitive baselines still favor PyTorch; profile before adding "
            "another broad launch-parameter sweep."
        )

    if _noisy_rows(rows):
        lines.append(
            "- Noisy rows should be profiled or rerun before treating their p50 latency as stable."
        )

    first_run = rows[0].run
    if not first_run.get("git_commit") or first_run.get("git_dirty") is None:
        lines.append(
            "- Source git metadata is missing; use `scripts/benchmark` or export "
            "`CUDA_KERNEL_LAB_GIT_COMMIT` and `CUDA_KERNEL_LAB_GIT_DIRTY` for SSH/tar runs."
        )

    return lines or ["- Use the fastest-by-operation table to choose the next profiler target."]


def _next_question(rows: list[ReportRow]) -> str:
    if _noisy_rows(rows):
        return "What does Nsight Compute show for the noisy Triton rows and the largest fused win?"

    memory_fastest = [row for row in _fastest_rows(rows) if row.primitive == "memory"]
    if memory_fastest and all(row.backend == "torch" for row in memory_fastest):
        return "What does Nsight Compute show for the Triton memory primitive bottleneck?"

    return "Which fastest Triton operation should be profiled next?"


def _top_triton_wins(rows: list[ReportRow]) -> list[ReportRow]:
    wins = [
        row
        for row in rows
        if row.backend == "triton" and row.speedup_vs_torch is not None and row.speedup_vs_torch > 1
    ]
    return sorted(wins, key=lambda row: row.speedup_vs_torch or 0, reverse=True)[:3]


def _noisy_rows(rows: list[ReportRow]) -> list[ReportRow]:
    noisy = [
        row
        for row in rows
        if row.noise_ratio is not None and row.noise_ratio >= NOISE_RATIO_THRESHOLD
    ]
    return sorted(noisy, key=lambda row: row.noise_ratio or 0, reverse=True)[:3]


def _dynamic_tail_rows(rows: list[ReportRow]) -> list[ReportRow]:
    return [row for row in rows if row.source.name == "decode-step-dynamic-tail.jsonl"]


def _worst_bucket_rows(
    rows: list[ReportRow],
) -> list[tuple[ReportRow, str, dict[str, Any]]]:
    worst_rows: list[tuple[ReportRow, str, dict[str, Any]]] = []
    for row in rows:
        worst = _worst_bucket(row)
        if worst is not None:
            bucket, metrics = worst
            worst_rows.append((row, bucket, metrics))
    return sorted(
        worst_rows,
        key=lambda item: _numeric_metric(item[2], "host_p95_ms") or 0.0,
        reverse=True,
    )


def _worst_bucket(row: ReportRow) -> tuple[str, dict[str, Any]] | None:
    breakdown = row.metrics.get("bucket_breakdown")
    if not isinstance(breakdown, dict):
        return None

    candidates: list[tuple[str, dict[str, Any]]] = []
    for bucket, metrics in breakdown.items():
        if isinstance(metrics, dict) and _numeric_metric(metrics, "host_p95_ms") is not None:
            candidates.append((str(bucket), metrics))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            _numeric_metric(item[1], "host_p95_ms") or 0.0,
            _numeric_metric(item[1], "host_tail_ratio_p95_p50") or 0.0,
        ),
    )


def _orchestration_rows(
    rows: list[ReportRow],
) -> list[tuple[ReportRow, str, dict[str, Any]]]:
    entries: list[tuple[ReportRow, str, dict[str, Any]]] = []
    for row in rows:
        breakdown = row.metrics.get("orchestration_breakdown")
        if not isinstance(breakdown, dict):
            continue
        for region, metrics in breakdown.items():
            if isinstance(metrics, dict):
                entries.append((row, str(region), metrics))
    return sorted(
        entries,
        key=lambda item: (
            _dynamic_row_sort_key(item[0]),
            item[1],
        ),
    )


def _dynamic_row_sort_key(row: ReportRow) -> tuple[str, str, int, str]:
    return (row.source.name, _bucket_policy_label(row), _seed_sort_value(row), row.strategy)


def _bucket_policy_label(row: ReportRow) -> str:
    buckets = row.parameters.get("batch_buckets")
    if isinstance(buckets, list | tuple):
        return ",".join(str(bucket) for bucket in buckets)

    args = row.run.get("args")
    if isinstance(args, dict):
        arg_buckets = args.get("batch_buckets")
        if arg_buckets is not None:
            return str(arg_buckets)
    return ""


def _seed_label(row: ReportRow) -> str:
    seed = _seed_value(row)
    return "" if seed is None else str(seed)


def _seed_sort_value(row: ReportRow) -> int:
    seed = _seed_value(row)
    return -1 if seed is None else seed


def _seed_value(row: ReportRow) -> int | None:
    args = row.run.get("args")
    if not isinstance(args, dict):
        return None
    seed = args.get("seed")
    if isinstance(seed, bool):
        return None
    if isinstance(seed, int):
        return seed
    return None


def _worst_bucket_label(value: tuple[str, dict[str, Any]] | None) -> str:
    if value is None:
        return ""
    bucket, metrics = value
    return f"{bucket} (p95 {_fmt_metric(metrics, 'host_p95_ms')} ms)"


def _fmt_metric(metrics: dict[str, Any], key: str) -> str:
    return _fmt_optional(_numeric_metric(metrics, key))


def _scheduler_p95_us(metrics: dict[str, Any]) -> float | None:
    value = _numeric_metric(metrics, "scheduler_decision_p95_us")
    if value is not None:
        return value
    return _numeric_metric(metrics, "scheduler_cpu_p95_us")


def _fmt_int_metric(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return ""
    return str(value)


def _numeric_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _mean(values: Any) -> float | None:
    numeric_values = [float(value) for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _counter_label(counter: Counter[str]) -> str:
    return ", ".join(f"{key} {count}" for key, count in sorted(counter.items()))


def _row_list_label(rows: list[ReportRow]) -> str:
    return "; ".join(_row_label(row) for row in rows)


def _row_label(row: ReportRow) -> str:
    label = f"{row.primitive} {row.operation} {row.dtype} {row.variant}"
    if row.speedup_vs_torch is not None and row.speedup_vs_torch > 1:
        return f"{label} ({_fmt(row.speedup_vs_torch)}x)"
    if row.noise_ratio is not None and row.noise_ratio >= NOISE_RATIO_THRESHOLD:
        return f"{label} ({_fmt(row.noise_ratio)} noise)"
    return label


def _technique_use_label(row: ReportRow) -> str:
    if row.backend == "torch":
        return "torch controls"
    return f"{row.primitive} {row.operation}"


def _used_by_label(values: list[str]) -> str:
    compacted = []
    for value in values:
        if value not in compacted:
            compacted.append(value)
    if len(compacted) <= 4:
        return ", ".join(compacted)
    return ", ".join(compacted[:4]) + f", +{len(compacted) - 4} more"


def _method_sort_key(family: str, method_id: str) -> tuple[int, str, str]:
    order = {
        "baseline": 0,
        "launch tuning": 1,
        "reduction": 2,
        "fusion": 3,
        "tiling": 4,
        "launch replay": 5,
    }
    return (order.get(family, 99), family, method_id)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def _primitive_label(benchmark: str) -> str:
    if benchmark == "memory_bandwidth":
        return "memory"
    return benchmark


def _strategy_label(backend: str) -> str:
    return "torch-baseline" if backend == "torch" else f"{backend}-kernel"


def _variant_label(run: dict[str, Any]) -> str:
    args = run.get("args")
    if not isinstance(args, dict):
        return "default"

    strategy_fields = {
        "memory_bandwidth": ("block_size",),
        "softmax": ("traffic_model",),
        "norms": ("eps",),
        "swiglu": ("block_size",),
        "matmul": ("block_m", "block_n", "block_k"),
        "attention": ("seq_len", "num_heads", "head_dim"),
        "decode_step": (
            "mode",
            "batch_size",
            "max_batch_size",
            "hidden_dim",
            "seq_len",
            "attention_backend",
            "dynamic_copy_mode",
            "piecewise_post_mode",
            "orchestration_timing",
        ),
    }
    fields = []
    for key in strategy_fields.get(str(run.get("benchmark")), ()):
        value = args.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return ", ".join(fields) if fields else "default"


def _correctness_label(value: Any) -> str:
    if not isinstance(value, dict) or not value.get("checked"):
        return "not checked"
    if value.get("passed") is True:
        return "pass"
    if value.get("passed") is False:
        return "fail"
    return "unknown"


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0 or not math.isfinite(denominator):
        return None
    return numerator / denominator


def _noise_label(noise_ratio: float | None) -> str:
    if noise_ratio is None:
        return ""
    label = _fmt(noise_ratio)
    if noise_ratio >= NOISE_RATIO_THRESHOLD:
        return f"{label} noisy"
    return label


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _fmt_optional(value: float | None) -> str:
    return "" if value is None else _fmt(value)


def _shape_label(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape)


def _nested(values: dict[str, Any], *keys: str) -> str:
    current: Any = values
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return "" if current is None else str(current)


def _cuda_devices_label(devices: Any) -> str:
    if not isinstance(devices, list) or not devices:
        return ""
    labels = []
    for device in devices:
        if isinstance(device, dict):
            name = device.get("name")
            total_memory = device.get("total_memory_bytes")
            if isinstance(total_memory, int):
                gib = total_memory / (1024**3)
                labels.append(f"`{name} ({gib:.2f} GiB)`")
            elif name:
                labels.append(f"`{name}`")
    return ", ".join(labels)


def _sort_key(row: ReportRow) -> tuple[str, str, str, tuple[int, ...], str, str]:
    return (row.primitive, row.operation, row.dtype, row.shape, row.variant, row.backend)


if __name__ == "__main__":
    main()

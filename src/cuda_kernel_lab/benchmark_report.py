"""Generate Markdown reports from benchmark JSONL records."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_INPUT_DIR = Path("experiments/results/aws-ec2-first-run")
DEFAULT_OUTPUT = Path("experiments/aws-ec2-first-gpu-run.md")
RESULT_FILES = ("memory.jsonl", "softmax.jsonl", "norms.jsonl")
NOISE_RATIO_THRESHOLD = 1.20


@dataclass(frozen=True)
class ReportRow:
    primitive: str
    operation: str
    backend: str
    dtype: str
    shape: tuple[int, ...]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    bandwidth_gbps: float
    tflops: float
    speedup_vs_torch: float | None
    noise_ratio: float | None
    run: dict[str, Any]
    source: Path


def load_report_rows(input_dir: Path) -> list[ReportRow]:
    """Load all benchmark rows from the standard first-run JSONL files."""

    rows: list[ReportRow] = []
    for filename in RESULT_FILES:
        path = input_dir / filename
        if not path.exists():
            continue
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
    """Render a compact first-run Markdown report."""

    if not rows:
        raise ValueError(f"no benchmark JSONL records found under {input_dir}")

    lines = [
        "# AWS EC2 First GPU Run",
        "",
        "Status: generated from benchmark JSONL",
        "",
        "## Question",
        "",
        "What are the first baseline PyTorch and Triton measurements for the CUDA Kernel",
        "Lab benchmark matrix on a disposable AWS EC2 `g5.xlarge` host in `us-west-2`?",
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
            "## Fastest By Operation",
            "",
            "| Primitive | Operation | Dtype | Shape | Fastest Backend | p50 ms | GB/s | TFLOP/s |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in _fastest_rows(rows):
        lines.append(
            "| "
            f"{row.primitive} | {row.operation} | {row.dtype} | {_shape_label(row.shape)} | "
            f"{row.backend} | {_fmt(row.p50_ms)} | {_fmt(row.bandwidth_gbps)} | "
            f"{_fmt(row.tflops)} |"
        )

    lines.extend(
        [
            "",
            "## Backend Detail",
            "",
            "| Primitive | Operation | Dtype | Shape | Backend | p50 ms | p95 ms | p99 ms | "
            "GB/s | TFLOP/s | Speedup vs Torch | Noise |",
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in sorted(rows, key=_sort_key):
        lines.append(
            "| "
            f"{row.primitive} | {row.operation} | {row.dtype} | {_shape_label(row.shape)} | "
            f"{row.backend} | {_fmt(row.p50_ms)} | {_fmt(row.p95_ms)} | {_fmt(row.p99_ms)} | "
            f"{_fmt(row.bandwidth_gbps)} | {_fmt(row.tflops)} | "
            f"{_fmt_optional(row.speedup_vs_torch)} | "
            f"{_noise_label(row.noise_ratio)} |"
        )

    lines.extend(
        [
            "",
            "## Observation",
            "",
            "Pending human interpretation.",
            "",
            "## Interpretation",
            "",
            "Pending profiler validation and strategy comparison.",
            "",
            "## Next Question",
            "",
            "Which memory-bandwidth strategy variant should be tested first for `vector_add`?",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Print report without writing it.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = load_report_rows(args.input_dir)
    report = render_markdown(rows, input_dir=args.input_dir)
    if args.dry_run:
        print(report)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}", file=sys.stderr)


def _row_from_record(record: dict[str, Any], *, source: Path, line_number: int) -> ReportRow:
    try:
        run = record["run"]
        result = record["result"]
        backend, operation = str(result["name"]).split(":", maxsplit=1)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid benchmark record in {source}:{line_number}") from exc

    return ReportRow(
        primitive=_primitive_label(str(run["benchmark"])),
        operation=operation,
        backend=backend,
        dtype=str(result["dtype"]),
        shape=tuple(int(dim) for dim in result["shape"]),
        p50_ms=float(result["p50_ms"]),
        p95_ms=float(result["p95_ms"]),
        p99_ms=float(result["p99_ms"]),
        bandwidth_gbps=float(result["bandwidth_gbps"]),
        tflops=float(result["tflops"]),
        speedup_vs_torch=None,
        noise_ratio=_ratio(float(result["p95_ms"]), float(result["p50_ms"])),
        run=run,
        source=source,
    )


def _with_speedups(rows: list[ReportRow]) -> list[ReportRow]:
    torch_p50 = {
        (row.primitive, row.operation, row.dtype, row.shape): row.p50_ms
        for row in rows
        if row.backend == "torch"
    }
    enriched = []
    for row in rows:
        baseline = torch_p50.get((row.primitive, row.operation, row.dtype, row.shape))
        speedup = _ratio(baseline, row.p50_ms) if baseline is not None else None
        enriched.append(
            ReportRow(
                primitive=row.primitive,
                operation=row.operation,
                backend=row.backend,
                dtype=row.dtype,
                shape=row.shape,
                p50_ms=row.p50_ms,
                p95_ms=row.p95_ms,
                p99_ms=row.p99_ms,
                bandwidth_gbps=row.bandwidth_gbps,
                tflops=row.tflops,
                speedup_vs_torch=speedup,
                noise_ratio=row.noise_ratio,
                run=row.run,
                source=row.source,
            )
        )
    return enriched


def _fastest_rows(rows: list[ReportRow]) -> list[ReportRow]:
    fastest: dict[tuple[str, str, str, tuple[int, ...]], ReportRow] = {}
    for row in rows:
        key = (row.primitive, row.operation, row.dtype, row.shape)
        current = fastest.get(key)
        if current is None or row.p50_ms < current.p50_ms:
            fastest[key] = row
    return sorted(fastest.values(), key=_sort_key)


def _primitive_label(benchmark: str) -> str:
    if benchmark == "memory_bandwidth":
        return "memory"
    return benchmark


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


def _sort_key(row: ReportRow) -> tuple[str, str, str, tuple[int, ...], str]:
    return (row.primitive, row.operation, row.dtype, row.shape, row.backend)


if __name__ == "__main__":
    main()
